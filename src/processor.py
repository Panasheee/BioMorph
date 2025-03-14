import cv2
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.morphology import skeletonize, medial_axis, thin
from skimage.measure import label, regionprops
from skimage import filters
from PIL import Image
import traceback
from sklearn.linear_model import RANSACRegressor
import traceback
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessor:
    """Handles all image processing operations with optimizations for microscopy images"""
    
    def __init__(self, config):
        self.config = config
        self.current_image = None
        self.original_image = None
        self.binary_mask = None
        self.processing_history = []
    
    def load_image(self, file_path):
        """Load an image from file with enhanced error handling"""
        try:
            print(f"Attempting to load image from: {file_path}")
            # Check if file exists first
            import os
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return False
                
            image = cv2.imread(file_path)
            if image is None:
                print(f"OpenCV returned None when reading: {file_path}")
                # Try using PIL as a fallback
                try:
                    from PIL import Image
                    pil_image = Image.open(file_path)
                    image = np.array(pil_image)
                    # Convert RGB to BGR for OpenCV if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    print("Successfully loaded image using PIL")
                except Exception as pil_error:
                    print(f"PIL also failed: {pil_error}")
                    raise ValueError(f"Could not read image with OpenCV or PIL: {file_path}")
                    
            print(f"Image loaded successfully: {image.shape}")
            self.current_image = image
            self.original_image = image.copy()
            self.processing_history = [("original", image.copy())]
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            traceback.print_exc()
            return False
    
    def prepare_binary_image(self, custom_params=None):
        """Convert current image to binary mask with enhanced preprocessing for microscopy images"""
        if self.current_image is None:
            print("No image loaded for binary conversion")
            return None
            
        # Use custom params if provided, otherwise use config
        params = custom_params or {
            "morph_open_size": self.config.get("morph_open_size", 3),
            "morph_close_size": self.config.get("morph_close_size", 3),
            "contrast_enhancement": True,
            "denoise": True
        }
        
        print("Preparing binary image with params:", params)
        
        # Convert to grayscale
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image.copy()  # Already grayscale
            
        # Apply contrast enhancement if enabled
        if params.get("contrast_enhancement"):
            print("Applying contrast enhancement")
            # Calculate histogram
            hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            # Normalize
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            # Create lookup table
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                if cdf[i] > 0:
                    lut[i] = np.round(cdf_normalized[i] / cdf_normalized.max() * 255)
            # Apply lookup table
            gray = cv2.LUT(gray, lut)
        
        # Apply denoising if enabled
        if params.get("denoise"):
            print("Applying denoising")
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply Gaussian blur to remove fine noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try adaptive thresholding first
        adaptive_threshold = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Also try Otsu thresholding
        _, otsu_threshold = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Use Canny edge detection to help with boundary detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Combine the methods
        combined = cv2.bitwise_or(adaptive_threshold, otsu_threshold)
        combined = cv2.bitwise_or(combined, edges)
        
        # Morphological operations for cleaning up
        if params["morph_open_size"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (params["morph_open_size"], params["morph_open_size"])
            )
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
        if params["morph_close_size"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (params["morph_close_size"], params["morph_close_size"])
            )
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
        # Store binary mask
        self.binary_mask = (combined > 0).astype(np.uint8)
        binary_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        self.processing_history.append(("binary", binary_bgr.copy()))
        
        print(f"Binary image prepared successfully. White pixel count: {np.sum(self.binary_mask)}")
        return binary_bgr
        
    def perform_voronoi_analysis(self):
        """
        Perform Voronoi diagram analysis optimized for microscopy images
        This is particularly useful for analyzing cellular or mycelial networks
        """
        if self.current_image is None:
            print("No image loaded for Voronoi analysis")
            return None
        
        print("Starting Voronoi analysis")
        
        try:
            # Make sure we have a binary mask
            if self.binary_mask is None:
                self.prepare_binary_image()
                
            if self.binary_mask is None or np.sum(self.binary_mask) < 10:
                print("Binary mask is empty or nearly empty")
                return None
                
            # Find contours on binary mask
            contours, _ = cv2.findContours(
                self.binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            print(f"Found {len(contours)} contours")
            
            # Extract points for Voronoi diagram
            # We'll use a combination of:
            # 1. Contour centroids (for large blobs)
            # 2. Branching points from skeletonization (for filament intersections)
            # 3. Random sampling of boundary points (for better tessellation)
            
            points = []
            
            # 1. Add centroids of significant contours
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10:  # Filter out noise
                    continue
                
                # Add centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append([cx, cy])
            
            # 2. Add branching points from skeleton
            # Get skeleton
            skeleton = skeletonize(self.binary_mask > 0)
            branch_points, _ = self._analyze_skeleton(skeleton)
            for point in branch_points:
                points.append([point[1], point[0]])  # Convert from (row, col) to (x, y)
            
            # 3. Add some boundary points by random sampling
            boundary = cv2.Canny(self.binary_mask * 255, 100, 200)
            boundary_points = np.where(boundary > 0)
            if len(boundary_points[0]) > 0:
                # Take up to 100 random points
                idx = np.random.choice(
                    len(boundary_points[0]), 
                    min(100, len(boundary_points[0])),
                    replace=False
                )
                for i in idx:
                    points.append([boundary_points[1][i], boundary_points[0][i]])
            
            if len(points) < 4:
                print("Not enough points for Voronoi analysis")
                return None
            
            # Add boundary points to ensure Voronoi diagram stays within image
            h, w = self.current_image.shape[:2]
            boundary_points = [
                [0, 0], [w//2, 0], [w-1, 0],
                [0, h//2], [w-1, h//2],
                [0, h-1], [w//2, h-1], [w-1, h-1]
            ]
            points.extend(boundary_points)
            
            print(f"Total points for Voronoi: {len(points)}")
            
            # Compute Voronoi diagram
            points = np.array(points)
            vor = Voronoi(points)
            
            # Create visualization
            result = self.current_image.copy()
            
            # Draw Voronoi regions
            region_areas = []
            for region in vor.regions:
                if not -1 in region and len(region) > 0:
                    polygon = []
                    valid = True
                    for idx in region:
                        vertex = vor.vertices[idx]
                        x, y = int(vertex[0]), int(vertex[1])
                        if 0 <= x < w and 0 <= y < h:
                            polygon.append([x, y])
                        else:
                            valid = False
                            break
                    
                    if valid and len(polygon) > 2:
                        polygon = np.array(polygon, dtype=np.int32)
                        area = cv2.contourArea(polygon)
                        region_areas.append(area)
                        
                        # Create more meaningful color - larger regions are darker
                        color_intensity = min(255, max(50, int(255 - area/100)))
                        color = (color_intensity, 
                                 min(200, color_intensity + np.random.randint(0, 50)), 
                                 color_intensity + np.random.randint(0, 30))
                        
                        cv2.fillPoly(result, [polygon], color, lineType=cv2.LINE_AA)
                        cv2.polylines(result, [polygon], True, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Blend with original image for better visualization
            alpha = 0.7
            beta = 0.3
            gamma = 0
            result = cv2.addWeighted(self.current_image, alpha, result, beta, gamma)
            
            # Calculate connectivity metrics
            metrics = {
                "num_regions": len(region_areas),
                "mean_area": np.mean(region_areas) if region_areas else 0,
                "std_area": np.std(region_areas) if region_areas else 0,
                "min_area": np.min(region_areas) if region_areas else 0, 
                "max_area": np.max(region_areas) if region_areas else 0,
                "connectivity_index": len(points) / (w * h) * 10000  # points per 10,000 pixels
            }
            
            self.processing_history.append(("voronoi", result.copy()))
            print("Voronoi analysis completed successfully")
            return result, metrics
            
        except Exception as e:
            print(f"Error in Voronoi analysis: {e}")
            traceback.print_exc()
            return None
        
    def perform_medial_axis_analysis(self):
        """
        Create skeleton/medial axis optimized for microscopy images
        This analysis focuses on the network structure of filaments
        """
        if self.current_image is None:
            print("No image loaded for medial axis analysis")
            return None
            
        print("Starting medial axis analysis")
        
        try:
            # Prepare binary image if not already done
            if self.binary_mask is None:
                self.prepare_binary_image()
                
            if self.binary_mask is None or np.sum(self.binary_mask) < 10:
                print("Binary mask is empty or nearly empty")
                return None
                
            # Try multiple skeletonization techniques and choose the best
            skeleton1 = skeletonize(self.binary_mask > 0)
            skeleton2 = thin(self.binary_mask > 0)  # Zhang-Suen algorithm
            
            # Also calculate distance transform for branch thickness
            dist_transform = cv2.distanceTransform(self.binary_mask * 255, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
            
            # Choose the skeleton that has more continuous lines
            if np.sum(skeleton1) >= np.sum(skeleton2):
                skeleton = skeleton1
                print("Using standard skeletonization")
            else:
                skeleton = skeleton2
                print("Using Zhang-Suen thinning")
            
            # Create RGB overlay
            overlay = self.current_image.copy()
            
            # Mark skeleton with coloring based on distance transform (branch thickness)
            skel_points = np.where(skeleton)
            for i in range(len(skel_points[0])):
                y, x = skel_points[0][i], skel_points[1][i]
                thickness = dist_transform[y, x]
                
                # Color based on thickness: thin branches in blue, thick branches in red
                color = (
                    int(255 * (1 - thickness)),  # B - higher for thin branches
                    0,                           # G - minimal
                    int(255 * thickness)         # R - higher for thick branches
                )
                
                cv2.circle(overlay, (x, y), 1, color, -1)
            
            # Calculate skeleton metrics
            branch_points, end_points = self._analyze_skeleton(skeleton)
            
            # Draw branch points and end points with more info
            for point in branch_points:
                # Yellow color for branch points, size indicates branching degree
                neighbors = self._count_neighbors(skeleton, point[0], point[1])
                radius = min(5, max(2, neighbors - 1))
                cv2.circle(overlay, (point[1], point[0]), radius, (0, 255, 255), -1)
                
            for point in end_points:
                # Magenta for end points
                cv2.circle(overlay, (point[1], point[0]), 3, (255, 0, 255), -1)
            
            # Calculate branches and connectivity metrics
            branch_lengths = self._calculate_branch_lengths(skeleton, branch_points, end_points)
            total_length = sum(branch_lengths)
            avg_branch_length = np.mean(branch_lengths) if branch_lengths else 0
            
            # Calculate area covered by the structures
            total_area = np.sum(self.binary_mask)
            
            # Calculate additional metrics for microscopy analysis
            metrics = {
                "skeleton_pixels": np.sum(skeleton),
                "branch_points": len(branch_points),
                "end_points": len(end_points),
                "total_branches": len(branch_lengths),
                "avg_branch_length": avg_branch_length,
                "total_skeleton_length": total_length,
                "branches_per_area": len(branch_points) / max(total_area, 1) * 1000,  # per 1000 pixels
                "network_complexity": len(branch_points) * avg_branch_length / max(total_area, 1) * 1000,
                "endpoint_ratio": len(end_points) / max(len(branch_points), 1)  # ratio of endpoints to branchpoints
            }
            
            self.processing_history.append(("medial_axis", overlay.copy()))
            print("Medial axis analysis completed successfully")
            return overlay, metrics
            
        except Exception as e:
            print(f"Error in medial axis analysis: {e}")
            traceback.print_exc()
            return None
            
    def _count_neighbors(self, skeleton, row, col):
        """Count the number of neighbors in a skeleton image"""
        height, width = skeleton.shape
        count = 0
        for i in range(max(0, row-1), min(height, row+2)):
            for j in range(max(0, col-1), min(width, col+2)):
                if (i != row or j != col) and skeleton[i, j]:
                    count += 1
        return count
    
    def _calculate_branch_lengths(self, skeleton, branch_points, end_points):
        """Calculate approximate lengths of branches in the skeleton"""
        # Label the skeleton
        labels = label(skeleton, connectivity=2)
        
        # Create a map of special points
        special_points = set()
        for point in branch_points + end_points:
            special_points.add((point[0], point[1]))
            
        # Create a copy of the skeleton
        skeleton_copy = skeleton.copy().astype(np.uint8)
        
        # Remove branch points to separate branches
        for point in branch_points:
            skeleton_copy[point[0], point[1]] = 0
        
        # Label the isolated branches
        branch_labels = label(skeleton_copy, connectivity=2)
        
        # Calculate branch lengths
        branch_lengths = []
        for region in regionprops(branch_labels):
            branch_lengths.append(region.area)
            
        return branch_lengths
    
    def _analyze_skeleton(self, skeleton):
        """Analyze skeleton to detect branch points and end points"""
        # Create a filter to detect junctions and endpoints
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # Center pixel doesn't count as neighbor
                           
        # Convolve with the kernel to count neighbors
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        skel_points = skeleton.astype(np.uint8)
        
        # Find branch points (more than 2 neighbors) and end points (1 neighbor)
        branch_points = np.where((neighbors > 2) & (skel_points > 0))
        end_points = np.where((neighbors == 1) & (skel_points > 0))
        
        # Convert to list of points
        branch_points = list(zip(branch_points[0], branch_points[1]))
        end_points = list(zip(end_points[0], end_points[1]))
        
        return branch_points, end_points
        
    def perform_fractal_analysis(self):
        """
        Calculate fractal dimension using box-counting method
        Enhanced for microscopy images to detect self-similarity patterns
        """
        if self.current_image is None:
            print("No image loaded for fractal analysis")
            return None
            
        print("Starting fractal analysis")
        
        try:
            # Ensure we have a binary mask
            if self.binary_mask is None:
                self.prepare_binary_image()
                
            if self.binary_mask is None or np.sum(self.binary_mask) < 10:
                print("Binary mask is empty or nearly empty")
                return None, None, None
            
            # Setup thresholds based on config
            steps = self.config.get("fractal_multi_threshold_steps", 3)
            
            # For microscopy images, use multiple thresholds to capture different structure levels
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Get Otsu threshold
            otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if steps <= 1:
                thresholds = [otsu_val]
            else:
                # Create thresholds around the Otsu value
                offset = 20  # Larger offset for more range
                half = steps // 2
                start = int(otsu_val - offset * half)
                thresholds = [max(0, start + offset * i) for i in range(steps)]
                
            print(f"Using {len(thresholds)} thresholds for fractal analysis: {thresholds}")
                
            results = []
            bin_img_final = None
            
            # For plotting
            all_log_sizes = []
            all_log_counts = []
            all_fds = []
            all_r2s = []
            
            for th in thresholds:
                # Apply threshold
                if np.mean(gray) > np.median(gray):  # Image is likely light background, dark features
                    _, bin_img = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY_INV)
                else:  # Image is likely dark background, light features
                    _, bin_img = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY)
                
                # Apply morphological operations
                morph_open_size = self.config.get("morph_open_size", 3)
                morph_close_size = self.config.get("morph_close_size", 3)
                
                if morph_open_size > 1:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)
                    )
                    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
                    
                if morph_close_size > 1:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)
                    )
                    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
                    
                bin_mask = (bin_img > 0).astype(np.uint8)
                
                # Skip if mask is empty
                if np.sum(bin_mask) < 10:
                    continue
                    
                bin_img_final = bin_img  # store last used for a visual
                
                # Calculate fractal dimension
                fract_dim, r2, log_sizes, log_counts = self._box_counting_fractal_dim(bin_mask)
                
                # Only include results with reasonable R² value
                if r2 > 0.8:  # Only include good fits
                    all_log_sizes.append(log_sizes)
                    all_log_counts.append(log_counts)
                    all_fds.append(fract_dim)
                    all_r2s.append(r2)
                    
                    results.append({
                        "threshold": th,
                        "fractal_dimension": fract_dim,
                        "r_squared": r2,
                        "log_sizes": log_sizes.tolist() if isinstance(log_sizes, np.ndarray) else log_sizes,
                        "log_counts": log_counts.tolist() if isinstance(log_counts, np.ndarray) else log_counts
                    })
            
            # Use the binary mask if no good threshold was found
            if bin_img_final is None:
                bin_img_final = self.binary_mask * 255
                
            # Create result image with color-coded fractal dimension
            if len(all_fds) > 0:
                # Create a visualization that highlights the fractal structures
                bin_final_bgr = self.current_image.copy()
                
                # Color the image based on the best fractal dimension
                best_idx = np.argmax(all_r2s) if all_r2s else 0
                best_fd = all_fds[best_idx] if all_fds else 1.0
                
                # Normalize FD to a color range: 1.0 (blue) to 2.0 (red)
                fd_normalized = min(1.0, max(0.0, (best_fd - 1.0) / 1.0))
                color = (
                    int(255 * (1 - fd_normalized)),  # B
                    0,                               # G
                    int(255 * fd_normalized)         # R
                )
                
                # Apply colorization to the binary mask
                mask = (bin_img_final > 0)
                bin_final_bgr[mask] = color
                
                # Add FD value as text
                cv2.putText(
                    bin_final_bgr,
                    f"FD: {best_fd:.3f} (R²: {all_r2s[best_idx]:.3f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            else:
                # Just convert to BGR if no good FD was found
                bin_final_bgr = cv2.cvtColor(bin_img_final, cv2.COLOR_GRAY2BGR)
            
            self.processing_history.append(("fractal", bin_final_bgr.copy()))
            
            # Calculate average metrics
            avg_fd = np.mean(all_fds) if all_fds else 0
            best_r2_idx = np.argmax(all_r2s) if all_r2s else 0
            best_fd = all_fds[best_r2_idx] if all_fds else 0
            
            metrics = {
                "average_fd": avg_fd,
                "best_fd": best_fd,
                "best_r2": all_r2s[best_r2_idx] if all_r2s else 0,
                "fractal_dimension_interpretation": self._interpret_fractal_dimension(best_fd),
                "threshold_values": thresholds,
                "fd_values": all_fds,
                "r2_values": all_r2s
            }
            
            print(f"Fractal analysis completed. Best FD: {best_fd:.4f}, R²: {metrics['best_r2']:.4f}")
            return bin_final_bgr, results, metrics
            
        except Exception as e:
            print(f"Error in fractal analysis: {e}")
            traceback.print_exc()
            return None, None, None
            
    def _interpret_fractal_dimension(self, fd):
        """Provide interpretation of the fractal dimension for biological images"""
        if fd < 1.2:
            return "Simple structure with minimal branching"
        elif fd < 1.4:
            return "Moderate complexity with some branching"
        elif fd < 1.6:
            return "Complex branching structure"
        elif fd < 1.8:
            return "Highly complex network with dense branching"
        else:
            return "Extremely complex space-filling structure"
            
    def _box_counting_fractal_dim(self, bin_mask):
        """Box-counting approach to calculate fractal dimension with improved accuracy"""
        def box_count(image_01, box_size):
            # Sum over sub-blocks
            S = np.add.reduceat(
                np.add.reduceat(image_01,
                               np.arange(0, image_01.shape[0], box_size), axis=0),
                np.arange(0, image_01.shape[1], box_size), axis=1
            )
            # Count boxes with sum > 0
            return np.count_nonzero(S)
            
        h, w = bin_mask.shape
        max_dim = min(h, w)
        
        # Use more box sizes for better regression
        # Include powers of 2, 3, and 5 for a more thorough analysis
        possible_box_sizes = []
        for i in range(1, int(np.log2(max_dim)) + 1):
            size = 2**i
            if size <= max_dim:
                possible_box_sizes.append(size)
                
        for i in range(1, int(np.log(max_dim)/np.log(3)) + 1):
            size = 3**i
            if size <= max_dim and size not in possible_box_sizes:
                possible_box_sizes.append(size)
                
        for i in range(1, int(np.log(max_dim)/np.log(1.5)) + 1):
            size = int(1.5**i)
            if size <= max_dim and size not in possible_box_sizes:
                possible_box_sizes.append(size)
                
        possible_box_sizes.sort()
        
        sizes = []
        counts = []
        for bs in possible_box_sizes:
            c = box_count(bin_mask, bs)
            if c > 0:
                sizes.append(bs)
                counts.append(c)
                
        if len(sizes) < 4:  # Need at least 4 points for reliable fit
            return 0.0, 0.0, [], []
            
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Use RANSAC for more robust fitting
        from sklearn.linear_model import RANSACRegressor
        
        # Reshape for sklearn
        X = log_sizes.reshape(-1, 1)
        y = log_counts.reshape(-1, 1)
        
        # Fit with RANSAC
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        slope = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]
        
        # Calculate fractal dimension
        fract_dim = -slope  # Negative slope => fractal dimension
        
        # Calculate R²
        predicted = slope * log_sizes + intercept
        ss_res = np.sum((log_counts - predicted)**2)
        ss_tot = np.sum((log_counts - np.mean(log_counts))**2)
        r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 1)
        
        return fract_dim, r2, log_sizes, log_counts
        
    def generate_histogram(self):
        """Generate histograms for BGR channels with compatibility fix"""
        if self.current_image is None:
            print("No image loaded for histogram generation")
            return None
            
        print("Generating histogram")
        
        try:
            # Create figure
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Calculate histograms
            if len(self.current_image.shape) == 3:  # Color image
                b, g, r = cv2.split(self.current_image)
                
                hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
                hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
                
                # Normalize histograms
                cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
                
                # Plot histograms with enhanced style
                ax.plot(hist_b, color='blue', alpha=0.7, label='Blue', linewidth=2)
                ax.plot(hist_g, color='green', alpha=0.7, label='Green', linewidth=2)
                ax.plot(hist_r, color='red', alpha=0.7, label='Red', linewidth=2)
                
                # Calculate metrics
                metrics = {
                    "mean_values": [float(np.mean(b)), float(np.mean(g)), float(np.mean(r))],
                    "std_values": [float(np.std(b)), float(np.std(g)), float(np.std(r))],
                    "median_values": [float(np.median(b)), float(np.median(g)), float(np.median(r))],
                    "min_values": [int(np.min(b)), int(np.min(g)), int(np.min(r))],
                    "max_values": [int(np.max(b)), int(np.max(g)), int(np.max(r))],
                    "brightness": float((np.mean(r) + np.mean(g) + np.mean(b)) / 3),
                    "contrast": float((np.std(r) + np.std(g) + np.std(b)) / 3)
                }
            else:  # Grayscale image
                gray = self.current_image
                
                hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist_gray, hist_gray, 0, 1, cv2.NORM_MINMAX)
                
                ax.plot(hist_gray, color='gray', alpha=0.9, label='Grayscale', linewidth=2)
                
                # Calculate metrics
                metrics = {
                    "mean_value": float(np.mean(gray)),
                    "std_value": float(np.std(gray)),
                    "median_value": float(np.median(gray)),
                    "min_value": int(np.min(gray)),
                    "max_value": int(np.max(gray)),
                    "brightness": float(np.mean(gray)),
                    "contrast": float(np.std(gray))
                }
                
            # Add labels and legend
            ax.set_title('Histogram Analysis')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Normalized Frequency')
            ax.legend()
            
            # Set background color
            fig.patch.set_facecolor('#121212')
            ax.set_facecolor('#121212')
            
            # Set text color to white
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Set grid and spines color
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            for spine in ax.spines.values():
                spine.set_color('gray')
            
            # Tight layout
            fig.tight_layout()
            
            # Draw the figure to a canvas
            canvas = FigureCanvasTkAgg(fig, master=None)
            canvas.draw()
            
            # FIX: Use the get_renderer method instead of tostring_rgb
            # Get the renderer
            renderer = canvas.get_renderer()
            # Get the width and height
            canvas_width, canvas_height = canvas.get_width_height()
            
            # Create a numpy array from the canvas
            img = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
            img = img.reshape((canvas_height, canvas_width, 4))  # RGBA format
            
            # Convert RGBA to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            self.processing_history.append(("histogram", img.copy()))
            print("Histogram generation completed successfully")
            return img, metrics
            
        except Exception as e:
            print(f"Error generating histogram: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def apply_equalization(self):
        """Apply advanced histogram equalization techniques"""
        if self.current_image is None:
            return None
            
        # Try CLAHE (Contrast Limited Adaptive Histogram Equalization) - better for microscopy
        try:
            if len(self.current_image.shape) == 3:
                # Convert to LAB color space for better perceptual enhancement
                lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # Merge channels
                enhanced_lab = cv2.merge((cl, a, b))
                
                # Convert back to BGR
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                # Apply CLAHE directly to grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(self.current_image)
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                
            self.processing_history.append(("enhanced", enhanced.copy()))
            return enhanced
        except Exception as e:
            print(f"Error applying equalization: {e}")
            traceback.print_exc()
            return None
            
    def apply_adaptive_thresholding(self):
        """Apply multi-scale adaptive thresholding optimized for microscopy"""
        if self.current_image is None:
            return None
            
        try:
            # Convert to grayscale if needed
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
                
            # Apply bilateral filter to preserve edges while removing noise
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Try different adaptive thresholding parameters
            adaptive1 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            adaptive2 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 3
            )
            
            # Combine the two adaptations
            combined = cv2.bitwise_and(adaptive1, adaptive2)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            # Convert to BGR for display
            adaptive_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            self.processing_history.append(("adaptive_threshold", adaptive_bgr.copy()))
            return adaptive_bgr
        except Exception as e:
            print(f"Error applying adaptive thresholding: {e}")
            traceback.print_exc()
            return None
    
    def apply_canny_edge(self):
        """Apply enhanced Canny edge detection with auto-parameters"""
        if self.current_image is None:
            return None
            
        try:
            # Convert to grayscale if needed
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
                
            # Apply gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Auto-detect parameters based on image statistics
            v = np.median(blur)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, lower, upper)
            
            # Dilate to connect nearby edges
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Convert to BGR for display
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            self.processing_history.append(("canny_edge", edges_bgr.copy()))
            return edges_bgr
        except Exception as e:
            print(f"Error applying Canny edge detection: {e}")
            traceback.print_exc()
            return None
    
    def reset_to_original(self):
        """Reset to the original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processing_history = [("original", self.original_image.copy())]
            self.binary_mask = None
            return self.original_image
        return None