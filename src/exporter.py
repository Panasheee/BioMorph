import os
import csv
import json
from datetime import datetime
from pathlib import Path
import cv2

class ResultExporter:
    """Handles exporting analysis results to files"""
    
    def __init__(self, config):
        self.config = config
        
    def get_export_directory(self, image_path=None):
        """Get directory for exporting results"""
        export_dir = self.config.get("export_directory")
        if not export_dir and image_path:
            export_dir = os.path.dirname(image_path)
        elif not export_dir:
            export_dir = str(Path.home())
        return export_dir
        
    def export_analysis_results(self, image_path, results):
        """Export all analysis results to a JSON file"""
        if not results:
            return False, ""
            
        export_dir = self.get_export_directory(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = "image"
            
        json_path = os.path.join(export_dir, f"{base_name}_analysis_{timestamp}.json")
        
        try:
            # Prepare exportable data (remove images)
            export_data = {}
            
            for analysis_type, data in results.items():
                if isinstance(data, dict):
                    export_data[analysis_type] = {}
                    for key, value in data.items():
                        if key != "image" and key != "details":
                            export_data[analysis_type][key] = value
                        elif key == "details":
                            # Include fractal details
                            export_data[analysis_type][key] = value
            
            # Add metadata
            export_data["metadata"] = {
                "timestamp": timestamp,
                "image": os.path.basename(image_path) if image_path else "unknown",
                "config": {
                    "morph_open_size": self.config.get("morph_open_size"),
                    "morph_close_size": self.config.get("morph_close_size"),
                    "fractal_steps": self.config.get("fractal_multi_threshold_steps")
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            return True, json_path
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False, ""
    
    def export_image(self, image, operation_name, image_path=None):
        """Export processed image to file"""
        if image is None:
            return False, ""
            
        try:
            export_dir = self.get_export_directory(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                base_name = "image"
                
            export_path = os.path.join(
                export_dir, 
                f"{base_name}_{operation_name}_{timestamp}.png"
            )
            
            cv2.imwrite(export_path, image)
            return True, export_path
        except Exception as e:
            print(f"Error exporting image: {e}")
            return False, ""
            
    def export_all_images(self, analysis_results, image_path=None):
        """Export all images from analysis results"""
        if not analysis_results:
            return []
            
        exported_paths = []
        
        for analysis_type, data in analysis_results.items():
            if isinstance(data, dict) and "image" in data:
                success, path = self.export_image(
                    data["image"], 
                    analysis_type, 
                    image_path
                )
                if success:
                    exported_paths.append((analysis_type, path))
                    
        return exported_paths
        
    def export_morphology_models(self, models, image_path=None):
        """Export all morphology models"""
        if not models:
            return []
            
        exported_paths = []
        
        for model_type, image in models.items():
            if image is not None:
                success, path = self.export_image(
                    image, 
                    f"morphology_{model_type}", 
                    image_path
                )
                if success:
                    exported_paths.append((model_type, path))
                    
        return exported_paths
        
    def export_full_report(self, analysis_results, morphology_models, image_path=None):
        """Export a comprehensive report with all results and images"""
        if not analysis_results:
            return False, ""
            
        # First export all images
        image_paths = self.export_all_images(analysis_results, image_path)
        model_paths = self.export_morphology_models(morphology_models, image_path)
        
        # Then export JSON results
        success, json_path = self.export_analysis_results(image_path, analysis_results)
        
        if not success:
            return False, ""
            
        # Create report file (HTML or Markdown)
        export_dir = self.get_export_directory(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = "image"
            
        report_path = os.path.join(export_dir, f"{base_name}_report_{timestamp}.html")
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Morphology Analysis Report - {base_name}</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            max-width: 1200px;
                            margin: 0 auto;
                            padding: 20px;
                            background-color: #121212;
                            color: #E0E0E0;
                        }}
                        h1, h2, h3 {{
                            color: #03DAC6;
                        }}
                        .section {{
                            margin-bottom: 30px;
                            padding: 20px;
                            background-color: #1E1E1E;
                            border-radius: 5px;
                        }}
                        .image-container {{
                            margin: 20px 0;
                            text-align: center;
                        }}
                        .image-container img {{
                            max-width: 100%;
                            border-radius: 5px;
                            border: 1px solid #333;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }}
                        th, td {{
                            padding: 12px 15px;
                            border-bottom: 1px solid #333;
                        }}
                        th {{
                            background-color: #333;
                        }}
                        .metrics {{
                            display: flex;
                            flex-wrap: wrap;
                            gap: 20px;
                        }}
                        .metric-box {{
                            background-color: #333;
                            padding: 10px;
                            border-radius: 5px;
                            min-width: 200px;
                        }}
                        .metric-value {{
                            font-size: 24px;
                            font-weight: bold;
                            color: #03DAC6;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Morphology Analysis Report</h1>
                    <div class="section">
                        <h2>Original Image</h2>
                        <p>File: {os.path.basename(image_path) if image_path else "Unknown"}</p>
                        <p>Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                """)
                
                # Add analysis results sections
                if "medial_axis" in analysis_results:
                    metrics = analysis_results["medial_axis"]["metrics"]
                    f.write(f"""
                    <div class="section">
                        <h2>Structural Analysis</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <p>Branch Points</p>
                                <p class="metric-value">{metrics.get("branch_points", "N/A")}</p>
                            </div>
                            <div class="metric-box">
                                <p>End Points</p>
                                <p class="metric-value">{metrics.get("end_points", "N/A")}</p>
                            </div>
                            <div class="metric-box">
                                <p>Branches per Area</p>
                                <p class="metric-value">{metrics.get("branches_per_area", "N/A"):.4f}</p>
                            </div>
                        </div>
                        
                        <div class="image-container">
                            <h3>Medial Axis Visualization</h3>
                            <img src="{os.path.basename(next((path for type, path in image_paths if type == 'medial_axis'), ''))}" alt="Medial Axis">
                        </div>
                    </div>
                    """)
                
                if "voronoi" in analysis_results:
                    metrics = analysis_results["voronoi"]["metrics"]
                    f.write(f"""
                    <div class="section">
                        <h2>Connectivity Analysis</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <p>Number of Regions</p>
                                <p class="metric-value">{metrics.get("num_regions", "N/A")}</p>
                            </div>
                            <div class="metric-box">
                                <p>Mean Region Area</p>
                                <p class="metric-value">{metrics.get("mean_area", "N/A"):.2f}</p>
                            </div>
                            <div class="metric-box">
                                <p>Area Std. Deviation</p>
                                <p class="metric-value">{metrics.get("std_area", "N/A"):.2f}</p>
                            </div>
                        </div>
                        
                        <div class="image-container">
                            <h3>Voronoi Visualization</h3>
                            <img src="{os.path.basename(next((path for type, path in image_paths if type == 'voronoi'), ''))}" alt="Voronoi">
                        </div>
                    </div>
                    """)
                
                if "fractal" in analysis_results:
                    metrics = analysis_results["fractal"]["metrics"]
                    f.write(f"""
                    <div class="section">
                        <h2>Complexity Analysis</h2>
                        <div class="metrics">
                            <div class="metric-box">
                                <p>Fractal Dimension</p>
                                <p class="metric-value">{metrics.get("best_fd", "N/A"):.4f}</p>
                            </div>
                            <div class="metric-box">
                                <p>Average FD</p>
                                <p class="metric-value">{metrics.get("average_fd", "N/A"):.4f}</p>
                            </div>
                            <div class="metric-box">
                                <p>R² Value</p>
                                <p class="metric-value">{metrics.get("best_r2", "N/A"):.4f}</p>
                            </div>
                        </div>
                        
                        <div class="image-container">
                            <h3>Binary Visualization</h3>
                            <img src="{os.path.basename(next((path for type, path in image_paths if type == 'fractal'), ''))}" alt="Fractal Binary">
                        </div>
                    </div>
                    """)
                
                # Add morphology models section
                if morphology_models:
                    f.write(f"""
                    <div class="section">
                        <h2>Computational Morphology Models</h2>
                        
                        <div class="image-container">
                            <h3>Structure Model</h3>
                            <img src="{os.path.basename(next((path for type, path in model_paths if type == 'phase1'), ''))}" alt="Structure Model">
                        </div>
                        
                        <div class="image-container">
                            <h3>Connectivity Model</h3>
                            <img src="{os.path.basename(next((path for type, path in model_paths if type == 'phase2'), ''))}" alt="Connectivity Model">
                        </div>
                        
                        <div class="image-container">
                            <h3>Complexity Model</h3>
                            <img src="{os.path.basename(next((path for type, path in model_paths if type == 'phase3'), ''))}" alt="Complexity Model">
                        </div>
                        
                        <div class="image-container">
                            <h3>Combined Model</h3>
                            <img src="{os.path.basename(next((path for type, path in model_paths if type == 'final'), ''))}" alt="Final Model">
                        </div>
                    </div>
                    """)
                
                # Add footer
                f.write(f"""
                    <div class="section">
                        <h2>Export Information</h2>
                        <p>JSON Data: <a href="{os.path.basename(json_path)}" style="color: #03DAC6">{os.path.basename(json_path)}</a></p>
                        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                </body>
                </html>
                """)
                
            return True, report_path
        except Exception as e:
            print(f"Error creating report: {e}")
            return False, ""