import cv2
import numpy as np
from scipy.spatial import Voronoi
import random
from skimage.morphology import skeletonize
import traceback

class MorphologyGenerator:
    """Generates computational models based on image analysis results
    Optimized for microscopic filamentous networks"""
    
    def __init__(self, analysis_results):
        self.results = analysis_results
        # Set standard dimensions for all generated models
        self.width, self.height = 800, 800
        
    def generate_structure_model(self):
        """
        Generate model representing basic structural features
        This model focuses on branching patterns and filament networks
        """
        try:
            print("Generating structure model")
            if "medial_axis" not in self.results:
                print("No medial axis data for structure model")
                return None
                
            # Get metrics from medial axis
            metrics = self.results["medial_axis"]["metrics"]
            
            # Create synthetic image based on key metrics
            model = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            model[:] = (15, 15, 15)  # Dark background for better contrast
            
            # Use branch points and end points to generate a structural visualization
            num_branches = metrics.get("branch_points", 20)
            num_endpoints = metrics.get("end_points", 40)
            
            # Calculate branch thickness based on metrics
            if "skeleton_pixels" in metrics and "branch_points" in metrics:
                avg_thickness = max(1.0, min(5.0, metrics["skeleton_pixels"] / max(metrics["branch_points"], 1) / 10))
            else:
                avg_thickness = 2.0
                
            # Normalize to reasonable ranges, scaled for larger model
            num_branches = min(max(10, num_branches), 100)
            num_endpoints = min(max(20, num_endpoints), 200)
            
            # Generate initial branch centers near the center of the image
            # Using a distribution that creates a more clustered, biological pattern
            center_x = self.width / 2
            center_y = self.height / 2
            radius = min(self.width, self.height) / 3
            
            # Create an initial branching structure
            branch_points = []
            for _ in range(num_branches // 3):  # Start with fewer main branches
                # Use polar coordinates for more natural distribution
                angle = np.random.uniform(0, 2 * np.pi)
                distance = radius * np.random.beta(2, 2)  # Beta distribution creates more natural clustering
                
                x = int(center_x + distance * np.cos(angle))
                y = int(center_y + distance * np.sin(angle))
                
                branch_points.append((x, y))
            
            # Function to draw a natural-looking branch with randomized thickness
            def draw_natural_branch(start_pt, end_pt, branch_level=0):
                # Calculate midpoint with some random displacement
                mid_x = (start_pt[0] + end_pt[0]) // 2
                mid_y = (start_pt[1] + end_pt[1]) // 2
                
                # Add random displacement, decreasing with branch level
                displacement = max(5, 30 // (branch_level + 1))
                mid_x += np.random.randint(-displacement, displacement)
                mid_y += np.random.randint(-displacement, displacement)
                
                # Keep within bounds
                mid_x = max(0, min(self.width-1, mid_x))
                mid_y = max(0, min(self.height-1, mid_y))
                
                # Calculate branch thickness, decreasing with branch level
                thickness = max(1, int(avg_thickness * (0.9 ** branch_level)))
                
                # Draw two line segments with randomized thickness
                cv2.line(model, start_pt, (mid_x, mid_y), (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.line(model, (mid_x, mid_y), end_pt, (255, 255, 255), thickness, cv2.LINE_AA)
                
                # Return midpoint
                return (mid_x, mid_y)
            
            # Draw recursive branches to create a fungal/hyphal-like network
            existing_endpoints = []
            
            # Draw initial branches from branch points
            for bx, by in branch_points:
                # Mark main branch point
                cv2.circle(model, (bx, by), 3, (0, 255, 255), -1)
                
                # Generate primary branches from this point
                num_branches_from_point = np.random.randint(3, 6)
                for _ in range(num_branches_from_point):
                    # Create a random length and angle
                    angle = np.random.uniform(0, 2 * np.pi)
                    length = np.random.randint(50, 150)
                    
                    # Calculate end point
                    ex = int(bx + length * np.cos(angle))
                    ey = int(by + length * np.sin(angle))
                    
                    # Ensure within bounds
                    ex = max(0, min(self.width-1, ex))
                    ey = max(0, min(self.height-1, ey))
                    
                    # Draw natural-looking branch
                    midpoint = draw_natural_branch((bx, by), (ex, ey))
                    
                    # Add branch point at midpoint
                    if np.random.random() < 0.7:  # 70% chance to add branch point
                        cv2.circle(model, midpoint, 2, (0, 255, 255), -1)
                        branch_points.append(midpoint)
                    
                    # Add endpoint
                    cv2.circle(model, (ex, ey), 2, (255, 0, 255), -1)
                    existing_endpoints.append((ex, ey))
            
            # Now add secondary branches from the newly added branch points
            secondary_branches = branch_points.copy()
            np.random.shuffle(secondary_branches)
            
            for level in range(3):  # Create multiple levels of branching
                new_branch_points = []
                
                for bx, by in secondary_branches[:min(len(secondary_branches), 30)]:
                    # Number of branches decreases with level
                    num_branches = np.random.randint(1, 4 - level)
                    
                    for _ in range(num_branches):
                        # Create a random length and angle
                        angle = np.random.uniform(0, 2 * np.pi)
                        length = np.random.randint(30, 100) // (level + 1)
                        
                        # Calculate end point
                        ex = int(bx + length * np.cos(angle))
                        ey = int(by + length * np.sin(angle))
                        
                        # Ensure within bounds
                        ex = max(0, min(self.width-1, ex))
                        ey = max(0, min(self.height-1, ey))
                        
                        # Draw natural-looking branch
                        midpoint = draw_natural_branch((bx, by), (ex, ey), branch_level=level+1)
                        
                        # Add branch point at midpoint with decreasing probability
                        if np.random.random() < 0.5 / (level + 1):
                            cv2.circle(model, midpoint, 2, (0, 255, 255), -1)
                            new_branch_points.append(midpoint)
                        
                        # Add endpoint
                        cv2.circle(model, (ex, ey), 2, (255, 0, 255), -1)
                        existing_endpoints.append((ex, ey))
                
                secondary_branches = new_branch_points
            
            # Create connections between some endpoints to form loops (common in fungal networks)
            endpoints = existing_endpoints.copy()
            np.random.shuffle(endpoints)
            
            num_connections = min(len(endpoints) // 4, 20)
            for i in range(num_connections):
                if i + 1 >= len(endpoints):
                    break
                    
                pt1 = endpoints[i]
                pt2 = endpoints[i + 1]
                
                # Only connect if within a reasonable distance
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if dist < 150:
                    # Draw natural-looking connecting branch
                    draw_natural_branch(pt1, pt2, branch_level=2)
            
            # Apply slight blur to make it look more natural
            model = cv2.GaussianBlur(model, (3, 3), 0)
            
            # Add title
            cv2.putText(
                model, 
                "Structural Morphology Model", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Add metrics text
            info_text = f"Branch Points: {metrics.get('branch_points', 'N/A')}"
            cv2.putText(
                model, 
                info_text, 
                (20, self.height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (200, 200, 200), 
                1, 
                cv2.LINE_AA
            )
            
            print("Structure model generated successfully")
            return model
            
        except Exception as e:
            print(f"Error generating structure model: {e}")
            traceback.print_exc()
            return None
        
    def generate_connectivity_model(self):
        """
        Generate model representing connectivity patterns in the network
        Optimized for fungal/filamentous networks
        """
        try:
            print("Generating connectivity model")
            
            # Check if we have voronoi data
            if "voronoi" not in self.results:
                # Try using medial axis data instead
                if "medial_axis" not in self.results:
                    print("No connectivity data available")
                    return None
                metrics = self.results["medial_axis"]["metrics"]
                region_count = metrics.get("branch_points", 30)
            else:
                # Use voronoi metrics
                metrics = self.results["voronoi"]["metrics"]
                region_count = metrics.get("num_regions", 30)
            
            # Create synthetic image based on key metrics
            model = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            model[:] = (15, 15, 15)  # Dark background
            
            # Calculate number of regions based on metrics
            num_regions = min(max(15, region_count), 100)
            
            print(f"Creating connectivity model with {num_regions} regions")
            
            # Generate points with clustering tendency (typical of biological networks)
            points = []
            
            # First, create some cluster centers
            num_clusters = min(num_regions // 5, 10)
            cluster_centers = []
            
            for _ in range(num_clusters):
                x = np.random.randint(100, self.width - 100)
                y = np.random.randint(100, self.height - 100)
                cluster_centers.append((x, y))
            
            # Generate points around cluster centers
            for _ in range(num_regions):
                # Select a random cluster center
                cx, cy = random.choice(cluster_centers)
                
                # Generate point with normal distribution around center
                std_dev = min(self.width, self.height) // 8
                x = int(np.random.normal(cx, std_dev))
                y = int(np.random.normal(cy, std_dev))
                
                # Ensure within bounds
                x = max(50, min(self.width - 50, x))
                y = max(50, min(self.height - 50, y))
                
                points.append([x, y])
                
            # Add boundary points
            boundary_points = [
                [0, 0], [self.width//3, 0], [2*self.width//3, 0], [self.width-1, 0],
                [0, self.height//3], [self.width-1, self.height//3],
                [0, 2*self.height//3], [self.width-1, 2*self.height//3],
                [0, self.height-1], [self.width//3, self.height-1], 
                [2*self.width//3, self.height-1], [self.width-1, self.height-1]
            ]
            points.extend(boundary_points)
            
            # Compute Voronoi diagram
            points = np.array(points)
            vor = Voronoi(points)
            
            # Draw Voronoi regions with biological-inspired coloring
            region_areas = []
            biological_colors = [
                (80, 100, 160),  # Bluish
                (80, 160, 120),  # Teal
                (120, 160, 80),  # Olive
                (160, 140, 80),  # Gold
                (160, 100, 80),  # Rust
                (140, 80, 120),  # Purple
                (100, 140, 140)  # Steel blue
            ]
            
            # Draw regions
            for region in vor.regions:
                if not -1 in region and len(region) > 0:
                    polygon = []
                    valid = True
                    for idx in region:
                        vertex = vor.vertices[idx]
                        x, y = int(vertex[0]), int(vertex[1])
                        if 0 <= x < self.width and 0 <= y < self.height:
                            polygon.append([x, y])
                        else:
                            valid = False
                            break
                    
                    if valid and len(polygon) > 2:
                        polygon = np.array(polygon, dtype=np.int32)
                        area = cv2.contourArea(polygon)
                        region_areas.append(area)
                        
                        # Create a biologically relevant color based on area
                        # Larger areas get darker colors
                        base_color = biological_colors[np.random.randint(0, len(biological_colors))]
                        area_factor = min(1.0, area / 20000) * 0.7
                        color = tuple([int(c * (1.0 - area_factor)) for c in base_color])
                        
                        cv2.fillPoly(model, [polygon], color, lineType=cv2.LINE_AA)
                        
                        # Draw edges in darker shade
                        edge_color = (min(c+30, 255) for c in color)
                        cv2.polylines(model, [polygon], True, tuple(edge_color), 1, cv2.LINE_AA)
            
            # Calculate mean region area
            mean_area = np.mean(region_areas) if region_areas else 0
            
            # Draw connections between regions - simulate hyphal connections
            # Extract all vertices from the Voronoi diagram
            vertices = []
            for region in vor.regions:
                if not -1 in region and len(region) > 0:
                    for idx in region:
                        vertex = vor.vertices[idx]
                        x, y = int(vertex[0]), int(vertex[1])
                        if 0 <= x < self.width and 0 <= y < self.height:
                            vertices.append((x, y))
            
            # Randomly connect some vertices to simulate hyphal bridges
            np.random.shuffle(vertices)
            num_connections = min(len(vertices) // 3, 40)
            
            for i in range(num_connections):
                if i + 1 >= len(vertices):
                    break
                    
                pt1 = vertices[i]
                pt2 = vertices[i + 1]
                
                # Only connect if within a reasonable distance
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if dist < 150:
                    # Draw connection with variable thickness
                    thickness = max(1, min(3, int(np.random.normal(1.5, 0.5))))
                    cv2.line(model, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Add title
            cv2.putText(
                model, 
                "Connectivity Morphology Model", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Add metrics text
            if "voronoi" in self.results and "metrics" in self.results["voronoi"]:
                region_info = f"Regions: {metrics.get('num_regions', 'N/A')}, "
                region_info += f"Avg Area: {metrics.get('mean_area', 0):.1f} px²"
                
                cv2.putText(
                    model, 
                    region_info, 
                    (20, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (200, 200, 200), 
                    1, 
                    cv2.LINE_AA
                )
            
            print("Connectivity model generated successfully")
            return model
            
        except Exception as e:
            print(f"Error generating connectivity model: {e}")
            traceback.print_exc()
            return None
        
    def generate_complexity_model(self):
        """
        Generate model representing complexity features
        Creates a fractal-like pattern based on the measured fractal dimension
        """
        try:
            print("Generating complexity model")
            if "fractal" not in self.results:
                print("No fractal data for complexity model")
                return None
                
            # Get metrics from fractal analysis
            metrics = self.results["fractal"]["metrics"]
            
            # Create synthetic image based on key metrics
            model = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            model[:] = (15, 15, 15)  # Dark background
            
            # Use fractal dimension to determine the complexity
            fd = metrics.get("best_fd", 1.5)
            
            # Normalize to reasonable range
            fd = min(max(1.1, fd), 2.0)
            
            # Map fractal dimension to iteration depth and branching factors
            iterations = int((fd - 1.0) * 7)
            iterations = min(max(2, iterations), 7)  # Cap at reasonable values
            
            # Higher FD means more branches and complexity
            branch_prob = min(0.9, 0.5 + (fd - 1.3) * 0.5)
            
            print(f"Creating fractal pattern with FD={fd:.3f}, iterations={iterations}, branch_prob={branch_prob:.2f}")
            
            # Generate fractal-like pattern appropriate for fungal/hyphal networks
            if fd < 1.4:
                # For lower FD, use a simpler branching pattern
                self._generate_simple_fractal(model, iterations, branch_prob)
            else:
                # For higher FD, use more complex pattern
                self._generate_complex_fractal(model, iterations, branch_prob)
            
            # Add title
            cv2.putText(
                model, 
                "Complexity Morphology Model", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Add fractal dimension
            fd_text = f"Fractal Dimension: {fd:.3f}"
            cv2.putText(
                model, 
                fd_text, 
                (20, self.height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (200, 200, 200), 
                1, 
                cv2.LINE_AA
            )
            
            print("Complexity model generated successfully")
            return model
            
        except Exception as e:
            print(f"Error generating complexity model: {e}")
            traceback.print_exc()
            return None
            
    def _generate_simple_fractal(self, image, iterations, branch_prob):
        """Generate a simple fractal pattern appropriate for fungal networks with lower complexity"""
        # Start with a simple trunk
        start_x = self.width // 2
        start_y = self.height * 3 // 4
        
        end_x = self.width // 2
        end_y = self.height // 4
        
        # Initial thickness based on iterations - thicker for fewer iterations
        thickness = max(1, 7 - iterations)
        
        # Draw initial trunk
        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Recursive function to draw branches
        def draw_branch(start_point, end_point, angle, length, thickness, iter_left):
            if iter_left <= 0 or thickness < 1 or length < 5:
                return
                
            # Draw the current branch
            cv2.line(image, start_point, end_point, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Calculate new length for sub-branches
            new_length = int(length * 0.7)
            
            # Calculate new angles for sub-branches
            left_angle = angle - np.random.uniform(20, 40)
            right_angle = angle + np.random.uniform(20, 40)
            
            # Calculate end points for new branches
            left_end_x = int(end_point[0] + new_length * np.sin(np.radians(left_angle)))
            left_end_y = int(end_point[1] - new_length * np.cos(np.radians(left_angle)))
            left_end = (left_end_x, left_end_y)
            
            right_end_x = int(end_point[0] + new_length * np.sin(np.radians(right_angle)))
            right_end_y = int(end_point[1] - new_length * np.cos(np.radians(right_angle)))
            right_end = (right_end_x, right_end_y)
            
            # Draw sub-branches with some randomness
            new_thickness = max(1, thickness - 1)
            
            # Always draw the main branches
            if iter_left > 1:
                # Draw left branch
                draw_branch(end_point, left_end, left_angle, new_length, new_thickness, iter_left - 1)
                
                # Draw right branch
                draw_branch(end_point, right_end, right_angle, new_length, new_thickness, iter_left - 1)
            
            # Randomly add extra branches based on branch probability
            if np.random.random() < branch_prob:
                # Calculate a random middle angle
                mid_angle = angle + np.random.uniform(-15, 15)
                mid_end_x = int(end_point[0] + new_length * np.sin(np.radians(mid_angle)))
                mid_end_y = int(end_point[1] - new_length * np.cos(np.radians(mid_angle)))
                mid_end = (mid_end_x, mid_end_y)
                
                # Draw middle branch
                draw_branch(end_point, mid_end, mid_angle, new_length, new_thickness, iter_left - 1)
        
        # Start the recursive drawing from the top of the trunk
        trunk_length = start_y - end_y
        draw_branch((start_x, start_y), (end_x, end_y), 0, trunk_length, thickness, iterations)
        
    def _generate_complex_fractal(self, image, iterations, branch_prob):
        """Generate a more complex fractal pattern for fungal networks with higher complexity"""
        # Multiple starting points for higher complexity
        num_starting_points = max(3, min(10, int(iterations * 1.5)))
        
        # Generate starting positions around the perimeter
        start_positions = []
        for _ in range(num_starting_points):
            side = np.random.randint(0, 4)
            if side == 0:  # Top
                x = np.random.randint(self.width // 4, 3 * self.width // 4)
                y = 0
                angle = np.random.uniform(45, 135)
            elif side == 1:  # Right
                x = self.width - 1
                y = np.random.randint(self.height // 4, 3 * self.height // 4)
                angle = np.random.uniform(135, 225)
            elif side == 2:  # Bottom
                x = np.random.randint(self.width // 4, 3 * self.width // 4)
                y = self.height - 1
                angle = np.random.uniform(225, 315)
            else:  # Left
                x = 0
                y = np.random.randint(self.height // 4, 3 * self.height // 4)
                angle = np.random.uniform(-45, 45)
            
            start_positions.append((x, y, angle))
        
        # Define the branching function
        def draw_complex_branch(x, y, angle, length, thickness, iter_left, color_offset=0):
            if iter_left <= 0 or thickness < 1 or length < 5:
                return
                
            # Calculate end point
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            
            # Ensure end point is within bounds
            end_x = max(0, min(self.width - 1, end_x))
            end_y = max(0, min(self.height - 1, end_y))
            
            # Determine color based on iteration level - deeper iterations get different colors
            if iter_left > iterations - 2:
                color = (200, 200, 200)  # White/gray for main branches
            else:
                # Mix colors based on iteration level
                r = min(255, max(50, 180 - iter_left * 20 + color_offset * 10))
                g = min(255, max(50, 180 - iter_left * 15 - color_offset * 5))
                b = min(255, max(50, 180 - iter_left * 10 - color_offset * 10))
                color = (b, g, r)
            
            # Draw this branch
            cv2.line(image, (x, y), (end_x, end_y), color, thickness, cv2.LINE_AA)
            
            # Calculate new parameters for sub-branches
            new_length = int(length * (0.6 + np.random.uniform(-0.1, 0.1)))
            
            # Number of sub-branches depends on iteration level and branch probability
            num_branches = 2  # Base case
            
            # Add more branches for higher complexity
            if iter_left > 1 and np.random.random() < branch_prob:
                num_branches += np.random.randint(1, 3)
            
            # Create sub-branches with distributed angles
            angle_range = min(120, 30 + iter_left * 10)  # Wider angle range for deeper iterations
            
            for i in range(num_branches):
                # Calculate angle for this sub-branch
                sub_angle = angle + np.random.uniform(-angle_range/2, angle_range/2)
                
                # Add some variation to length and thickness
                sub_length = new_length * np.random.uniform(0.8, 1.2)
                sub_thickness = max(1, thickness - 1)
                
                # Draw the sub-branch
                draw_complex_branch(end_x, end_y, sub_angle, sub_length, sub_thickness, 
                                   iter_left - 1, color_offset + i)
        
        # Draw from each starting position
        for x, y, angle in start_positions:
            # Initial branch length proportional to image size
            initial_length = min(self.width, self.height) / (num_starting_points * 0.8)
            
            # Initial thickness based on iterations
            initial_thickness = max(1, 5 - iterations//2)
            
            # Start recursive drawing
            draw_complex_branch(x, y, angle, initial_length, initial_thickness, iterations)
        
        # Add some connecting branches between existing structures
        # First, find points where branches exist by checking white pixels
        white_points = np.where(np.any(image > 100, axis=2))
        if len(white_points[0]) > 0:
            # Sample some points
            idx = np.random.choice(len(white_points[0]), min(50, len(white_points[0])), replace=False)
            points = [(white_points[1][i], white_points[0][i]) for i in idx]
            
            # Create some connections
            for i in range(min(20, len(points)-1)):
                p1 = points[i]
                p2 = points[i+1]
                
                # Only connect if within reasonable distance
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if 50 < dist < 200:
                    cv2.line(image, p1, p2, (100, 150, 200), 1, cv2.LINE_AA)
        
    def generate_combined_model(self):
        """
        Generate final combined morphological model that integrates 
        structural, connectivity and complexity features
        """
        try:
            print("Generating combined morphology model")
            
            # Get individual models
            structure = self.generate_structure_model()
            connectivity = self.generate_connectivity_model()
            complexity = self.generate_complexity_model()
            
            if structure is None and connectivity is None and complexity is None:
                print("No valid models to combine")
                return None
                
            # Create a combined visualization
            final_model = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            final_model[:] = (15, 15, 15)  # Dark background
            
            # Apply a layered approach for better visualization
            
            # Layer 1: Start with complexity pattern as the background if available
            if complexity is not None:
                # Blend with reduced opacity
                alpha = 0.4
                final_model = cv2.addWeighted(final_model, 1-alpha, complexity, alpha, 0)
            
            # Layer 2: Add connectivity regions if available
            if connectivity is not None:
                # Create a mask for non-black pixels in connectivity model
                conn_gray = cv2.cvtColor(connectivity, cv2.COLOR_BGR2GRAY)
                _, conn_mask = cv2.threshold(conn_gray, 20, 255, cv2.THRESH_BINARY)
                
                # Blend connectivity where it exists
                alpha = 0.5
                for y in range(self.height):
                    for x in range(self.width):
                        if conn_mask[y, x] > 0:
                            final_model[y, x] = cv2.addWeighted(
                                np.array([final_model[y, x]], dtype=np.uint8),
                                1-alpha,
                                np.array([connectivity[y, x]], dtype=np.uint8),
                                alpha,
                                0
                            )
            
            # Layer 3: Add structural elements on top with highest visibility
            if structure is not None:
                # Get structural elements (white lines and colored points)
                struct_gray = cv2.cvtColor(structure, cv2.COLOR_BGR2GRAY)
                _, struct_mask = cv2.threshold(struct_gray, 50, 255, cv2.THRESH_BINARY)
                
                # Apply structural elements with full opacity 
                alpha = 0.8
                for y in range(self.height):
                    for x in range(self.width):
                        if struct_mask[y, x] > 0:
                            final_model[y, x] = cv2.addWeighted(
                                np.array([final_model[y, x]], dtype=np.uint8),
                                1-alpha,
                                np.array([structure[y, x]], dtype=np.uint8),
                                alpha,
                                0
                            )
                            
                # Ensure branch and end points are fully visible
                # Extract yellow (branch points) and magenta (end points)
                yellow_mask = cv2.inRange(structure, (0, 240, 240), (15, 255, 255))
                magenta_mask = cv2.inRange(structure, (240, 0, 240), (255, 15, 255))
                
                # Apply these with full opacity
                for y in range(self.height):
                    for x in range(self.width):
                        if yellow_mask[y, x] > 0:
                            final_model[y, x] = (0, 255, 255)  # Yellow
                        elif magenta_mask[y, x] > 0:
                            final_model[y, x] = (255, 0, 255)  # Magenta
            
            # Add highlights and finishing touches
            
            # Apply slight blur for smoother appearance
            final_model = cv2.GaussianBlur(final_model, (3, 3), 0)
            
            # Add title with shadow for better visibility
            title = "Integrated Morphological Model"
            
            # Shadow
            cv2.putText(
                final_model, 
                title, 
                (22, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 0), 
                3, 
                cv2.LINE_AA
            )
            
            # Main text
            cv2.putText(
                final_model, 
                title, 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Add metrics from all three analyses
            metrics_text = ""
            
            if "medial_axis" in self.results and "metrics" in self.results["medial_axis"]:
                metrics = self.results["medial_axis"]["metrics"]
                if "branch_points" in metrics:
                    metrics_text += f"Branch Points: {metrics['branch_points']}  "
                    
            if "fractal" in self.results and "metrics" in self.results["fractal"]:
                metrics = self.results["fractal"]["metrics"]
                if "best_fd" in metrics:
                    metrics_text += f"FD: {metrics['best_fd']:.3f}  "
                    
            if "voronoi" in self.results and "metrics" in self.results["voronoi"]:
                metrics = self.results["voronoi"]["metrics"]
                if "num_regions" in metrics:
                    metrics_text += f"Regions: {metrics['num_regions']}"
            
            # Add metrics text with shadow
            if metrics_text:
                # Shadow
                cv2.putText(
                    final_model, 
                    metrics_text, 
                    (22, self.height - 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), 
                    2, 
                    cv2.LINE_AA
                )
                
                # Main text
                cv2.putText(
                    final_model, 
                    metrics_text, 
                    (20, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (200, 200, 255), 
                    1, 
                    cv2.LINE_AA
                )
            
            print("Combined morphology model generated successfully")
            return final_model
            
        except Exception as e:
            print(f"Error generating combined model: {e}")
            traceback.print_exc()
            return None