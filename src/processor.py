import cv2
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.morphology import skeletonize
from PIL import Image

class ImageProcessor:
    """Handles all image processing operations"""
    
    def __init__(self, config):
        self.config = config
        self.current_image = None
        self.original_image = None
        self.binary_mask = None
        self.processing_history = []
    
    def load_image(self, file_path):
        """Load an image from file"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read image")
            self.current_image = image
            self.original_image = image.copy()
            self.processing_history = [("original", image.copy())]
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def prepare_binary_image(self, custom_params=None):
        """Convert current image to binary mask using configurable parameters"""
        if self.current_image is None:
            return None
            
        # Use custom params if provided, otherwise use config
        params = custom_params or {
            "morph_open_size": self.config.get("morph_open_size"),
            "morph_close_size": self.config.get("morph_close_size")
        }
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        if params["morph_open_size"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (params["morph_open_size"], params["morph_open_size"])
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
        if params["morph_close_size"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (params["morph_close_size"], params["morph_close_size"])
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
        self.binary_mask = (binary > 0).astype(np.uint8)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.processing_history.append(("binary", binary_bgr.copy()))
        return binary_bgr
        
    def perform_voronoi_analysis(self):
        """Perform Voronoi diagram analysis on binary image"""
        if self.current_image is None:
            return None
            
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract points for Voronoi diagram
        points = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:  # Filter out very small contours
                continue
            
            # Add centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])
        
        if len(points) < 4:
            return None
        
        # Add boundary points to ensure Voronoi diagram stays within image
        h, w = self.current_image.shape[:2]
        boundary_points = [
            [0, 0], [w//2, 0], [w-1, 0],
            [0, h//2], [w-1, h//2],
            [0, h-1], [w//2, h-1], [w-1, h-1]
        ]
        points.extend(boundary_points)
        
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
                    
                    color = tuple(np.random.randint(50, 200, 3).tolist())  # Softer colors
                    cv2.fillPoly(result, [polygon], color, lineType=cv2.LINE_AA)
                    cv2.polylines(result, [polygon], True, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Blend with original image
        result = cv2.addWeighted(self.current_image, 0.7, result, 0.3, 0)
        
        # Store metrics
        metrics = {
            "num_regions": len(region_areas),
            "mean_area": np.mean(region_areas) if region_areas else 0,
            "std_area": np.std(region_areas) if region_areas else 0,
            "min_area": np.min(region_areas) if region_areas else 0, 
            "max_area": np.max(region_areas) if region_areas else 0
        }
        
        self.processing_history.append(("voronoi", result.copy()))
        return result, metrics
        
    def perform_medial_axis_analysis(self):
        """Create skeleton/medial axis and overlay on original image"""
        if self.current_image is None:
            return None
            
        # Prepare binary image if not already done
        if self.binary_mask is None:
            self.prepare_binary_image()
            
        if self.binary_mask is None:
            return None
            
        # Generate skeleton
        skeleton = skeletonize(self.binary_mask > 0)
        
        # Create RGB overlay
        overlay = self.current_image.copy()
        skeleton_uint8 = skeleton.astype(np.uint8) * 255
        
        # Mark skeleton in red
        overlay[skeleton, 0] = 0    # B
        overlay[skeleton, 1] = 0    # G
        overlay[skeleton, 2] = 255  # R
        
        # Calculate skeleton metrics
        branch_points, end_points = self._analyze_skeleton(skeleton)
        
        # Draw branch points and end points
        for point in branch_points:
            cv2.circle(overlay, (point[1], point[0]), 3, (0, 255, 255), -1)  # Yellow for branch points
            
        for point in end_points:
            cv2.circle(overlay, (point[1], point[0]), 3, (255, 0, 255), -1)  # Magenta for end points
        
        # Calculate metrics
        metrics = {
            "skeleton_pixels": np.sum(skeleton),
            "branch_points": len(branch_points),
            "end_points": len(end_points),
            "branches_per_area": len(branch_points) / max(np.sum(self.binary_mask), 1) * 1000  # per 1000 pixels
        }
        
        self.processing_history.append(("medial_axis", overlay.copy()))
        return overlay, metrics
    
    def _analyze_skeleton(self, skeleton):
        """Analyze skeleton to detect branch points and end points"""
        # Create a filter to detect junctions (more than 2 neighbors)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
                           
        # Convolve with the kernel to count neighbors
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        neighbors = neighbors * skeleton.astype(np.uint8)
        
        # Find branch points (more than 2 neighbors) and end points (1 neighbor)
        branch_points = np.where(neighbors > 2)
        end_points = np.where(neighbors == 1)
        
        # Convert to list of points
        branch_points = list(zip(branch_points[0], branch_points[1]))
        end_points = list(zip(end_points[0], end_points[1]))
        
        return branch_points, end_points
        
    def perform_fractal_analysis(self):
        """Calculate fractal dimension using box-counting method"""
        if self.current_image is None:
            return None, None
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Get Otsu threshold
        otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Setup thresholds based on config
        steps = self.config.get("fractal_multi_threshold_steps")
        if steps <= 1:
            thresholds = [otsu_val]
        else:
            offset = 10
            half = steps // 2
            start = int(otsu_val - offset * half)
            thresholds = [max(0, start + offset * i) for i in range(steps)]
            
        results = []
        bin_img_final = None
        
        # For plotting
        all_log_sizes = []
        all_log_counts = []
        all_fds = []
        all_r2s = []
        
        for th in thresholds:
            _, bin_img = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            morph_open_size = self.config.get("morph_open_size")
            morph_close_size = self.config.get("morph_close_size")
            
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
            bin_img_final = bin_img  # store last used for a visual
            
            # Calculate fractal dimension
            fract_dim, r2, log_sizes, log_counts = self._box_counting_fractal_dim(bin_mask)
            
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
        
        # Create result image
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
            "threshold_values": thresholds,
            "fd_values": all_fds,
            "r2_values": all_r2s
        }
        
        return bin_final_bgr, results, metrics
        
    def _box_counting_fractal_dim(self, bin_mask):
        """Box-counting approach to calculate fractal dimension"""
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
        possible_box_sizes = [2**i for i in range(1, 12) if 2**i <= max_dim]
        
        sizes = []
        counts = []
        for bs in possible_box_sizes:
            c = box_count(bin_mask, bs)
            if c > 0:
                sizes.append(bs)
                counts.append(c)
                
        if len(sizes) < 2:
            return 0.0, 0.0, [], []
            
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Linear fit
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        fract_dim = -slope  # Negative slope => fractal dimension
        
        # Calculate R²
        predicted = slope * log_sizes + intercept
        ss_res = np.sum((log_counts - predicted)**2)
        ss_tot = np.sum((log_counts - np.mean(log_counts))**2)
        r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 1)
        
        return fract_dim, r2, log_sizes, log_counts
        
    def generate_histogram(self):
        """Generate histograms for BGR channels"""
        if self.current_image is None:
            return None
            
        b, g, r = cv2.split(self.current_image)
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate histograms
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        
        # Plot histograms
        ax.plot(hist_b, color='blue', alpha=0.7, label='Blue')
        ax.plot(hist_g, color='green', alpha=0.7, label='Green')
        ax.plot(hist_r, color='red', alpha=0.7, label='Red')
        
        # Add labels and legend
        ax.set_title('Histogram')
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
        
        # Calculate some metrics
        metrics = {
            "mean_values": [np.mean(b), np.mean(g), np.mean(r)],
            "std_values": [np.std(b), np.std(g), np.std(r)],
            "median_values": [np.median(b), np.median(g), np.median(r)],
            "min_values": [np.min(b), np.min(g), np.min(r)],
            "max_values": [np.max(b), np.max(g), np.max(r)]
        }
        
        # Tight layout
        fig.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        self.processing_history.append(("histogram", img.copy()))
        return img, metrics
        
    def apply_equalization(self):
        """Apply histogram equalization to the image"""
        if self.current_image is None:
            return None
            
        img_yuv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        self.processing_history.append(("equalized", equalized.copy()))
        return equalized
        
    def apply_adaptive_thresholding(self):
        """Apply adaptive thresholding to the image"""
        if self.current_image is None:
            return None
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        
        self.processing_history.append(("adaptive_threshold", adaptive_bgr.copy()))
        return adaptive_bgr
        
    def apply_canny_edge(self):
        """Apply Canny edge detection to the image"""
        if self.current_image is None:
            return None
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        median_val = np.median(blur)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        
        edges = cv2.Canny(blur, lower, upper)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        self.processing_history.append(("canny_edge", edges_bgr.copy()))
        return edges_bgr
        
    def reset_to_original(self):
        """Reset to the original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.processing_history = [("original", self.original_image.copy())]
            self.binary_mask = None
            return self.original_image
        return None