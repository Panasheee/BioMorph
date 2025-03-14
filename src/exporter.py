import os
import json
from datetime import datetime
from pathlib import Path
import cv2
import traceback
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

class ResultExporter:
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
        
        base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else "image"
        json_path = os.path.join(export_dir, f"{base_name}_analysis_{timestamp}.json")
        
        try:
            export_data = {}
            for analysis_type, data in results.items():
                if isinstance(data, dict):
                    export_data[analysis_type] = {}
                    for key, value in data.items():
                        if key not in ("image", "details"):
                            export_data[analysis_type][key] = self._convert_np_types(value)
                        elif key == "details":
                            export_data[analysis_type][key] = self._convert_np_types(value)
            
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
            traceback.print_exc()
            return False, ""

    def _convert_np_types(self, data):
        """Recursively convert NumPy types to native Python types"""
        import numpy as np
        if isinstance(data, dict):
            return {k: self._convert_np_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_np_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def export_image(self, image, operation_name, image_path=None):
        """Export processed image to file"""
        if image is None:
            return False, ""
        try:
            export_dir = self.get_export_directory(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else "image"
            export_path = os.path.join(export_dir, f"{base_name}_{operation_name}_{timestamp}.png")
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
                success, path = self.export_image(data["image"], analysis_type, image_path)
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
                success, path = self.export_image(image, f"morphology_{model_type}", image_path)
                if success:
                    exported_paths.append((model_type, path))
        return exported_paths

    def export_pdf_report(self, analysis_results, morphology_models, image_path=None):
        """Export a comprehensive PDF report with all results and images"""
        if not analysis_results:
            return False, ""
        try:
            export_dir = self.get_export_directory(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else "image"
            report_path = os.path.join(export_dir, f"{base_name}_report_{timestamp}.pdf")
            
            # Set up ReportLab document
            doc = SimpleDocTemplate(report_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph("Morphology Analysis Report", styles['Title']))
            story.append(Spacer(1, 20))
            
            # Original image (if available)
            if image_path:
                story.append(Paragraph(f"Original Image: {os.path.basename(image_path)}", styles['Heading2']))
                try:
                    orig_img = RLImage(image_path, width=400, height=300)
                    story.append(orig_img)
                except Exception as e:
                    story.append(Paragraph("Error displaying original image.", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Structural analysis section
            if "medial_axis" in analysis_results:
                story.append(Paragraph("Structural Analysis", styles['Heading2']))
                metrics = analysis_results["medial_axis"].get("metrics", {})
                for key, value in metrics.items():
                    story.append(Paragraph(f"{key}: {value}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Connectivity analysis section
            if "voronoi" in analysis_results:
                story.append(Paragraph("Connectivity Analysis", styles['Heading2']))
                metrics = analysis_results["voronoi"].get("metrics", {})
                for key, value in metrics.items():
                    story.append(Paragraph(f"{key}: {value}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Complexity analysis section
            if "fractal" in analysis_results:
                story.append(Paragraph("Complexity Analysis", styles['Heading2']))
                metrics = analysis_results["fractal"].get("metrics", {})
                for key, value in metrics.items():
                    story.append(Paragraph(f"{key}: {value}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Morphology models section
            model_paths = self.export_morphology_models(morphology_models, image_path)
            if model_paths:
                story.append(Paragraph("Computational Morphology Models", styles['Heading2']))
                for model_type, model_path in model_paths:
                    story.append(Paragraph(f"Model: {model_type}", styles['Heading3']))
                    try:
                        model_img = RLImage(model_path, width=400, height=300)
                        story.append(model_img)
                    except Exception as e:
                        story.append(Paragraph("Error including model image.", styles['Normal']))
                    story.append(Spacer(1, 20))
            
            # Footer with generation timestamp
            story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            
            # Build PDF document
            doc.build(story)
            return True, report_path
        except Exception as e:
            print(f"Error creating PDF report: {e}")
            traceback.print_exc()
            return False, ""
    
    def export_full_report(self, analysis_results, morphology_models, image_path=None):
        """Export a comprehensive report as a PDF."""
        return self.export_pdf_report(analysis_results, morphology_models, image_path)