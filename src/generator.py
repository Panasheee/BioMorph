import cv2
import numpy as np
from scipy.spatial import Voronoi
import random

class MorphologyGenerator:
    """Generates computational models based on image analysis results"""
    
    def __init__(self, analysis_results):
        self.results = analysis_results
        
    def generate_structure_model(self):
        """Generate model representing basic structural features"""
        if "medial_axis" not in self.results:
            return None
            
        # Get metrics from medial axis
        metrics = self.results["medial_axis"]["metrics"]
        
        # Create synthetic image based on key metrics
        width, height = 600, 600
        model = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use branch points and end points to generate a structural visualization
        num_branches = metrics.get("branch_points", 10)
        num_endpoints = metrics.get("end_points", 20)
        
        # Normalize to reasonable ranges
        num_branches = min(max(5, num_branches), 30)
        num_endpoints = min(max(10, num_endpoints), 50)
        
        # Generate random points for branch centers
        branch_points = []
        for _ in range(num_branches):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            branch_points.append((x, y))
            
        # Draw branches
        for bx, by in branch_points:
            # Draw main branch point
            cv2.circle(model, (bx, by), 3, (0, 255, 255), -1)
            
            # Generate branches from this point
            num_branches_from_point = np.random.randint(2, 6)
            for _ in range(num_branches_from_point):
                # Random angle and length
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(30, 100)
                
                # Calculate end point
                ex = int(bx + length * np.cos(angle))
                ey = int(by + length * np.sin(angle))
                
                # Ensure within bounds
                ex = max(0, min(width-1, ex))
                ey = max(0, min(height-1, ey))
                
                # Draw branch
                cv2.line(model, (bx, by), (ex, ey), (255, 255, 255), 1)
                
                # Add endpoint
                cv2.circle(model, (ex, ey), 2, (255, 0, 255), -1)
        
        return model
        
    def generate_connectivity_model(self):
        """Generate model representing connectivity patterns"""
        if "voronoi" not in self.results:
            return None
            
        # Get metrics from Voronoi analysis
        metrics = self.results["voronoi"]["metrics"]
        
        # Create synthetic image based on key metrics
        width, height = 600, 600
        model = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use Voronoi region metrics to generate visualization
        num_regions = metrics.get("num_regions", 20)
        mean_area = metrics.get("mean_area", 1000)
        
        # Normalize to reasonable ranges
        num_regions = min(max(10, num_regions), 50)
        
        # Generate random points
        points = []
        for _ in range(num_regions):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            points.append([x, y])
            
        # Add boundary points
        boundary_points = [
            [0, 0], [width//2, 0], [width-1, 0],
            [0, height//2], [width-1, height//2],
            [0, height-1], [width//2, height-1], [width-1, height-1]
        ]
        points.extend(boundary_points)
        
        # Compute Voronoi diagram
        points = np.array(points)
        vor = Voronoi(points)
        
        # Draw Voronoi regions
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                polygon = []
                valid = True
                for idx in region:
                    vertex = vor.vertices[idx]
                    x, y = int(vertex[0]), int(vertex[1])
                    if 0 <= x < width and 0 <= y < height:
                        polygon.append([x, y])
                    else:
                        valid = False
                        break
                
                if valid and len(polygon) > 2:
                    polygon = np.array(polygon, dtype=np.int32)
                    color = tuple(np.random.randint(50, 200, 3).tolist())
                    cv2.fillPoly(model, [polygon], color, lineType=cv2.LINE_AA)
                    cv2.polylines(model, [polygon], True, (255, 255, 255), 1, cv2.LINE_AA)
        
        return model
        
    def generate_complexity_model(self):
        """Generate model representing complexity features"""
        if "fractal" not in self.results:
            return None
            
        # Get metrics from fractal analysis
        metrics = self.results["fractal"]["metrics"]
        
        # Create synthetic image based on key metrics
        width, height = 600, 600
        model = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use fractal dimension to determine the complexity
        fd = metrics.get("best_fd", 1.5)
        
        # Normalize to reasonable range
        fd = min(max(1.1, fd), 2.0)
        
        # Map fractal dimension to iteration depth
        iterations = int((fd - 1.0) * 10)
        iterations = min(max(1, iterations), 6)  # Cap at reasonable values
        
        # Generate fractal-like pattern (simplified Koch curve-inspired)
        self._generate_fractal_pattern(model, iterations)
        
        return model
        
    def _generate_fractal_pattern(self, image, iterations):
        """Draw a fractal-like pattern on the image"""
        height, width = image.shape[:2]
        
        # Start with a simple shape (e.g., triangle)
        p1 = (width//4, height*3//4)
        p2 = (width//2, height//4)
        p3 = (width*3//4, height*3//4)
        
        # Initial triangle
        points = [p1, p2, p3]
        
        # Draw the fractal
        self._recursive_fractal(image, points, iterations)
        
    def _recursive_fractal(self, image, points, iterations, color_offset=0):
        """Recursively draw fractal pattern"""
        if iterations <= 0:
            # Draw the final shape
            pts = np.array(points, dtype=np.int32)
            color = ((50 + color_offset) % 255, 
                    (100 + color_offset * 2) % 255, 
                    (150 + color_offset * 3) % 255)
            cv2.fillPoly(image, [pts], color)
            cv2.polylines(image, [pts], True, (255, 255, 255), 1)
            return
            
        # Subdivide each edge and create new points
        new_points = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Calculate midpoint
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            
            # Calculate perpendicular point (for added complexity)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Perpendicular vector
            perp_x = -dy
            perp_y = dx
            
            # Normalize and scale
            length = np.sqrt(perp_x**2 + perp_y**2)
            if length > 0:
                perp_x = int(perp_x / length * np.random.randint(10, 20))
                perp_y = int(perp_y / length * np.random.randint(10, 20))
                
                # New point perpendicular to edge
                new_x = mid_x + perp_x
                new_y = mid_y + perp_y
                
                # Ensure within bounds
                h, w = image.shape[:2]
                new_x = max(0, min(w-1, new_x))
                new_y = max(0, min(h-1, new_y))
                
                new_points.extend([p1, (new_x, new_y)])
            else:
                new_points.append(p1)
        
        # Recursive call with new set of points
        self._recursive_fractal(image, new_points, iterations - 1, color_offset + 40)
        
    def generate_combined_model(self):
        """Generate final combined morphological model"""
        # Get individual models
        structure = self.generate_structure_model()
        connectivity = self.generate_connectivity_model()
        complexity = self.generate_complexity_model()
        
        if structure is None or connectivity is None or complexity is None:
            return None
            
        # Create a combined visualization
        width, height = 600, 600
        final_model = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Blend the three models with different weights
        if structure is not None:
            final_model = cv2.addWeighted(final_model, 0.7, structure, 0.3, 0)
        
        if connectivity is not None:
            mask = cv2.cvtColor(connectivity, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8) * 255
            mask_inv = cv2.bitwise_not(mask)
            
            # Extract regions from connectivity
            conn_masked = cv2.bitwise_and(connectivity, connectivity, mask=mask)
            final_masked = cv2.bitwise_and(final_model, final_model, mask=mask_inv)
            final_model = cv2.add(final_masked, conn_masked)
        
        if complexity is not None:
            # Apply complexity pattern as overlay with transparency
            alpha = 0.3
            for y in range(height):
                for x in range(width):
                    if np.any(complexity[y, x] > 0):
                        final_model[y, x] = cv2.addWeighted(
                            np.array([final_model[y, x]], dtype=np.uint8), 
                            1-alpha,
                            np.array([complexity[y, x]], dtype=np.uint8), 
                            alpha, 
                            0
                        )
        
        # Add title
        cv2.putText(
            final_model, 
            "Computational Morphology Model", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        return final_model