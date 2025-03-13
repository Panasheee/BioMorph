import cv2
import numpy as np

class AnalysisPipeline:
    """Manages the full analysis pipeline and computational generation"""
    
    def __init__(self, processor, exporter, config):
        self.processor = processor
        self.exporter = exporter
        self.config = config
        self.analysis_results = {}
        self.morphology_models = {}
        
def run_full_analysis(self, image_path=None):
    """Run all analyses in sequence and collect results"""
    # If image path is provided, load it
    if image_path:
        success = self.processor.load_image(image_path)
        if not success:
            return False
            
    # Check if we have an image loaded
    if self.processor.current_image is None:
        return False
        
    # Store results from each analysis
    results = {}
    
    # 1. Perform binary image preparation
    binary_image = self.processor.prepare_binary_image()
    results["binary"] = {"image": binary_image}
    
    # 2. Run Voronoi analysis - add error handling
    voronoi_result = self.processor.perform_voronoi_analysis()
    if voronoi_result is None:
        results["voronoi"] = {"image": None, "metrics": {}}
    else:
        voronoi_image, voronoi_metrics = voronoi_result
        results["voronoi"] = {
            "image": voronoi_image,
            "metrics": voronoi_metrics
        }
    
    # 3. Run Medial Axis analysis - add error handling
    medial_result = self.processor.perform_medial_axis_analysis()
    if medial_result is None:
        results["medial_axis"] = {"image": None, "metrics": {}}
    else:
        medial_image, medial_metrics = medial_result
        results["medial_axis"] = {
            "image": medial_image,
            "metrics": medial_metrics
        }
    
    # 4. Run Fractal analysis - add error handling
    fractal_result = self.processor.perform_fractal_analysis()
    if fractal_result is None or len(fractal_result) < 3:
        results["fractal"] = {"image": None, "details": [], "metrics": {}}
    else:
        fractal_image, fractal_results, fractal_metrics = fractal_result
        results["fractal"] = {
            "image": fractal_image,
            "details": fractal_results,
            "metrics": fractal_metrics
        }
    
    # 5. Generate histogram - add error handling
    hist_result = self.processor.generate_histogram()
    if hist_result is None:
        results["histogram"] = {"image": None, "metrics": {}}
    else:
        hist_image, hist_metrics = hist_result
        results["histogram"] = {
            "image": hist_image,
            "metrics": hist_metrics
        }
    
    # Store all results
    self.analysis_results = results
    
    # Generate morphological models if enabled
    if self.config.get("generate_models"):
        self.generate_morphology_models()
        
    return True
        
    def generate_morphology_models(self):
        """Generate computational models based on analysis results"""
        if not self.analysis_results:
            return
            
        # Create morphology generator - fix the import issue
        # Import here to avoid circular imports
        from generator import MorphologyGenerator
        generator = MorphologyGenerator(self.analysis_results)
        
        # Generate models for each phase
        self.morphology_models = {
            "phase1": generator.generate_structure_model(),
            "phase2": generator.generate_connectivity_model(),
            "phase3": generator.generate_complexity_model(),
            "final": generator.generate_combined_model()
        }
        
        return self.morphology_models