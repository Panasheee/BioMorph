import os
import csv
import json
from datetime import datetime
from pathlib import Path
import cv2
import traceback
import os
import csv
import json
from datetime import datetime
from pathlib import Path
import cv2

class ResultExporter:

    # Enhanced HTML report generation method for ResultExporter class

    def export_full_report(self, analysis_results, morphology_models, image_path=None):
        """Export a comprehensive report with all results and images - optimized for fungal/filamentous networks"""
        if not analysis_results:
            return False, ""
            
        try:
            # First export all images
            image_paths = self.export_all_images(analysis_results, image_path)
            model_paths = self.export_morphology_models(morphology_models, image_path)
            
            # Then export JSON results for data preservation
            success, json_path = self.export_analysis_results(image_path, analysis_results)
            
            if not success:
                return False, ""
                
            # Create report file (HTML)
            export_dir = self.get_export_directory(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                base_name = "image"
                
            report_path = os.path.join(export_dir, f"{base_name}_report_{timestamp}.html")
            
            # Get measurements from results for the report
            measurements = {}
            
            # Structure measurements
            if "medial_axis" in analysis_results and "metrics" in analysis_results["medial_axis"]:
                metrics = analysis_results["medial_axis"]["metrics"]
                measurements["structure"] = {
                    "branch_points": metrics.get("branch_points", "N/A"),
                    "end_points": metrics.get("end_points", "N/A"),
                    "branches_per_area": metrics.get("branches_per_area", "N/A"),
                    "avg_branch_length": metrics.get("avg_branch_length", "N/A"),
                    "network_complexity": metrics.get("network_complexity", "N/A"),
                    "endpoint_ratio": metrics.get("endpoint_ratio", "N/A")
                }
            
            # Connectivity measurements
            if "voronoi" in analysis_results and "metrics" in analysis_results["voronoi"]:
                metrics = analysis_results["voronoi"]["metrics"]
                measurements["connectivity"] = {
                    "num_regions": metrics.get("num_regions", "N/A"),
                    "mean_area": metrics.get("mean_area", "N/A"),
                    "std_area": metrics.get("std_area", "N/A"),
                    "connectivity_index": metrics.get("connectivity_index", "N/A")
                }
            
            # Complexity measurements
            if "fractal" in analysis_results and "metrics" in analysis_results["fractal"]:
                metrics = analysis_results["fractal"]["metrics"]
                measurements["complexity"] = {
                    "best_fd": metrics.get("best_fd", "N/A"),
                    "best_r2": metrics.get("best_r2", "N/A"),
                    "average_fd": metrics.get("average_fd", "N/A"),
                    "fractal_dimension_interpretation": metrics.get("fractal_dimension_interpretation", "")
                }
                
            # Generate interpretations based on measurements
            interpretations = {}
            
            # Structure interpretation
            if "structure" in measurements:
                m = measurements["structure"]
                
                branch_density = m["branches_per_area"]
                if isinstance(branch_density, (int, float)):
                    if branch_density < 0.5:
                        branching = "low branching density indicates a simple network with minimal complexity"
                    elif branch_density < 2.0:
                        branching = "moderate branching density suggests a balanced network structure"
                    else:
                        branching = "high branching density indicates a complex, highly interconnected network"
                else:
                    branching = "branching density could not be determined"
                    
                endpoint_ratio = m["endpoint_ratio"]
                if isinstance(endpoint_ratio, (int, float)):
                    if endpoint_ratio < 1.0:
                        endpoints = "low endpoint ratio suggests a closed or circular network structure"
                    elif endpoint_ratio < 2.0:
                        endpoints = "balanced endpoint ratio indicates a mix of closed and open pathways"
                    else:
                        endpoints = "high endpoint ratio suggests an open network with many terminating branches"
                else:
                    endpoints = "endpoint ratio could not be determined"
                    
                interpretations["structure"] = f"The network shows {branching}. The {endpoints}."
            
            # Connectivity interpretation
            if "connectivity" in measurements:
                m = measurements["connectivity"]
                
                num_regions = m["num_regions"]
                if isinstance(num_regions, (int, float)):
                    if num_regions < 10:
                        regions = "low number of regions suggests a sparse network with limited connectivity"
                    elif num_regions < 50:
                        regions = "moderate number of regions indicates a well-developed network structure"
                    else:
                        regions = "high number of regions suggests a dense, highly connected network"
                else:
                    regions = "number of regions could not be determined"
                    
                mean_area = m["mean_area"]
                std_area = m["std_area"]
                if isinstance(mean_area, (int, float)) and isinstance(std_area, (int, float)) and mean_area > 0:
                    cv = std_area / mean_area
                    if cv < 0.5:
                        variation = "low variation in region sizes indicates uniform network distribution"
                    elif cv < 1.0:
                        variation = "moderate variation in region sizes suggests a balanced network with some specialized areas"
                    else:
                        variation = "high variation in region sizes indicates heterogeneous network with some dominant structural elements"
                else:
                    variation = "variation in region sizes could not be determined"
                    
                interpretations["connectivity"] = f"The {regions}. The {variation}."
            
            # Complexity interpretation
            if "complexity" in measurements:
                m = measurements["complexity"]
                
                if "fractal_dimension_interpretation" in m and m["fractal_dimension_interpretation"]:
                    interpretations["complexity"] = m["fractal_dimension_interpretation"]
                else:
                    fd = m["best_fd"]
                    if isinstance(fd, (int, float)):
                        if fd < 1.2:
                            complexity = "low fractal dimension indicates a simple structure with minimal branching"
                        elif fd < 1.4:
                            complexity = "moderate fractal dimension suggests a structure with some branching and complexity"
                        elif fd < 1.6:
                            complexity = "high fractal dimension indicates a complex network with significant branching"
                        elif fd < 1.8:
                            complexity = "very high fractal dimension suggests a highly complex, space-filling structure"
                        else:
                            complexity = "extremely high fractal dimension indicates an exceptionally complex structure approaching plane-filling"
                    else:
                        complexity = "fractal dimension could not be determined"
                        
                    interpretations["complexity"] = f"The {complexity}."
            
            # Create HTML report with modern design
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>BioMorph Analysis: {base_name}</title>
                    <style>
                        :root {{
                            --bg-primary: #121212;
                            --bg-secondary: #1E1E1E;
                            --bg-elevated: #2D2D2D;
                            --accent: #03DAC6;
                            --accent-secondary: #BB86FC;
                            --text-primary: #E0E0E0;
                            --text-secondary: #A0A0A0;
                        }}
                        
                        body {{
                            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                            line-height: 1.6;
                            max-width: 1200px;
                            margin: 0 auto;
                            padding: 20px;
                            background-color: var(--bg-primary);
                            color: var(--text-primary);
                        }}
                        
                        h1, h2, h3, h4 {{
                            color: var(--accent);
                            font-weight: 600;
                        }}
                        
                        h1 {{
                            font-size: 2.5rem;
                            margin-bottom: 1rem;
                            border-bottom: 2px solid var(--accent);
                            padding-bottom: 0.5rem;
                        }}
                        
                        h2 {{
                            font-size: 1.8rem;
                            margin-top: 2rem;
                            margin-bottom: 1rem;
                        }}
                        
                        h3 {{
                            font-size: 1.4rem;
                            margin-top: 1.5rem;
                            margin-bottom: 0.8rem;
                            color: var(--accent-secondary);
                        }}
                        
                        .section {{
                            margin-bottom: 40px;
                            padding: 25px;
                            background-color: var(--bg-secondary);
                            border-radius: 8px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                        }}
                        
                        .image-container {{
                            margin: 25px 0;
                            text-align: center;
                        }}
                        
                        .image-container img {{
                            max-width: 100%;
                            border-radius: 8px;
                            border: 1px solid #333;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                        }}
                        
                        .image-container figcaption {{
                            margin-top: 10px;
                            color: var(--text-secondary);
                            font-style: italic;
                        }}
                        
                        .image-grid {{
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                            gap: 20px;
                            margin: 25px 0;
                        }}
                        
                        .metrics-grid {{
                            display: grid;
                            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                            gap: 20px;
                            margin: 25px 0;
                        }}
                        
                        .metric-box {{
                            background-color: var(--bg-elevated);
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 3px solid var(--accent);
                        }}
                        
                        .metric-title {{
                            font-size: 0.9rem;
                            color: var(--text-secondary);
                            margin-bottom: 5px;
                        }}
                        
                        .metric-value {{
                            font-size: 1.8rem;
                            font-weight: bold;
                            color: var(--accent);
                            margin-bottom: 5px;
                        }}
                        
                        .metric-unit {{
                            font-size: 0.8rem;
                            color: var(--text-secondary);
                        }}
                        
                        .interpretation {{
                            background-color: var(--bg-elevated);
                            padding: 20px;
                            border-radius: 8px;
                            margin: 20px 0;
                            border-left: 3px solid var(--accent-secondary);
                        }}
                        
                        .interpretation-title {{
                            color: var(--accent-secondary);
                            font-weight: bold;
                            margin-bottom: 10px;
                        }}
                        
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }}
                        
                        th, td {{
                            padding: 12px 15px;
                            border-bottom: 1px solid #333;
                            text-align: left;
                        }}
                        
                        th {{
                            background-color: var(--bg-elevated);
                            color: var(--accent);
                        }}
                        
                        .header {{
                            display: flex;
                            align-items: center;
                            justify-content: space-between;
                            margin-bottom: 30px;
                        }}
                        
                        .header-info {{
                            max-width: 60%;
                        }}
                        
                        .header-logo {{
                            font-size: 2.5rem;
                            font-weight: bold;
                            color: var(--accent);
                        }}
                        
                        .header-subtitle {{
                            color: var(--text-secondary);
                            font-size: 1.2rem;
                        }}
                        
                        .footer {{
                            margin-top: 50px;
                            padding-top: 20px;
                            border-top: 1px solid #333;
                            color: var(--text-secondary);
                            font-size: 0.9rem;
                            text-align: center;
                        }}
                        
                        .tag {{
                            display: inline-block;
                            background-color: var(--bg-elevated);
                            color: var(--accent);
                            padding: 5px 10px;
                            border-radius: 4px;
                            margin-right: 5px;
                            font-size: 0.8rem;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <div class="header-info">
                            <div class="header-logo">BioMorph</div>
                            <div class="header-subtitle">Advanced Morphological Analysis Report</div>
                        </div>
                        <div>
                            <span class="tag">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Analysis Overview</h2>
                        <p>
                            <strong>File:</strong> {os.path.basename(image_path) if image_path else "Unknown"}<br>
                            <strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </p>
                        
                        <div class="image-container">
                            <img src="{os.path.basename(next((path for type, path in image_paths if type == 'binary'), ''))}" 
                                alt="Binary Segmentation">
                            <figcaption>Binary segmentation of the analyzed structure</figcaption>
                        </div>
                        
                        <div class="interpretation">
                            <div class="interpretation-title">Summary Interpretation</div>
                            <p>This analysis examines the morphological characteristics of the filamentous network structure, focusing on three key aspects:</p>
                            <ul>
                                <li><strong>Structure:</strong> {interpretations.get('structure', 'No structural interpretation available.')}</li>
                                <li><strong>Connectivity:</strong> {interpretations.get('connectivity', 'No connectivity interpretation available.')}</li>
                                <li><strong>Complexity:</strong> {interpretations.get('complexity', 'No complexity interpretation available.')}</li>
                            </ul>
                        </div>
                    </div>
                """)
                
                # Add structure analysis section
                if "medial_axis" in analysis_results and "metrics" in analysis_results["medial_axis"]:
                    metrics = analysis_results["medial_axis"]["metrics"]
                    medial_axis_path = next((path for type, path in image_paths if type == 'medial_axis'), '')
                    
                    if medial_axis_path:
                        f.write(f"""
                        <div class="section">
                            <h2>Structural Analysis</h2>
                            <p>
                                Structural analysis examines the branching patterns, junctions, and endpoints 
                                of the filamentous network. These features reveal the basic architectural 
                                organization of the network.
                            </p>
                            
                            <div class="image-container">
                                <img src="{os.path.basename(medial_axis_path)}" alt="Structural Analysis">
                                <figcaption>Medial Axis visualization showing branch points (yellow) and end points (magenta)</figcaption>
                            </div>
                            
                            <h3>Structural Metrics</h3>
                            <div class="metrics-grid">
                        """)
                        
                        if "branch_points" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Branch Points</div>
                                <div class="metric-value">{metrics["branch_points"]}</div>
                                <div class="metric-unit">Junction nodes in the network</div>
                            </div>
                            """)
                            
                        if "end_points" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">End Points</div>
                                <div class="metric-value">{metrics["end_points"]}</div>
                                <div class="metric-unit">Terminal nodes in the network</div>
                            </div>
                            """)
                            
                        if "branches_per_area" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Branch Density</div>
                                <div class="metric-value">{metrics["branches_per_area"]:.2f}</div>
                                <div class="metric-unit">Branches per 1000 px²</div>
                            </div>
                            """)
                        
                        if "avg_branch_length" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Average Branch Length</div>
                                <div class="metric-value">{metrics["avg_branch_length"]:.2f}</div>
                                <div class="metric-unit">pixels</div>
                            </div>
                            """)
                        
                        if "endpoint_ratio" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Endpoint/Branch Ratio</div>
                                <div class="metric-value">{metrics["endpoint_ratio"]:.2f}</div>
                                <div class="metric-unit">ratio</div>
                            </div>
                            """)
                            
                        if "network_complexity" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Network Complexity</div>
                                <div class="metric-value">{metrics["network_complexity"]:.2f}</div>
                                <div class="metric-unit">complexity index</div>
                            </div>
                            """)
                            
                        f.write("""
                            </div> <!-- End metrics-grid -->
                        """)
                        
                        if "structure" in interpretations:
                            f.write(f"""
                            <div class="interpretation">
                                <div class="interpretation-title">Structural Interpretation</div>
                                <p>{interpretations["structure"]}</p>
                            </div>
                            """)
                            
                        f.write("""
                        </div> <!-- End section -->
                        """)
                
                # Add connectivity analysis section
                if "voronoi" in analysis_results and "metrics" in analysis_results["voronoi"]:
                    metrics = analysis_results["voronoi"]["metrics"]
                    voronoi_path = next((path for type, path in image_paths if type == 'voronoi'), '')
                    
                    if voronoi_path:
                        f.write(f"""
                        <div class="section">
                            <h2>Connectivity Analysis</h2>
                            <p>
                                Connectivity analysis examines the spatial relationships and regional organization
                                of the network. This reveals how different parts of the network are interconnected
                                and how space is partitioned.
                            </p>
                            
                            <div class="image-container">
                                <img src="{os.path.basename(voronoi_path)}" alt="Connectivity Analysis">
                                <figcaption>Voronoi diagram showing regional partitioning of the network</figcaption>
                            </div>
                            
                            <h3>Connectivity Metrics</h3>
                            <div class="metrics-grid">
                        """)
                        
                        if "num_regions" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Number of Regions</div>
                                <div class="metric-value">{metrics["num_regions"]}</div>
                                <div class="metric-unit">distinct spatial regions</div>
                            </div>
                            """)
                            
                        if "mean_area" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Mean Region Area</div>
                                <div class="metric-value">{metrics["mean_area"]:.2f}</div>
                                <div class="metric-unit">pixels²</div>
                            </div>
                            """)
                            
                        if "std_area" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Region Size Variation</div>
                                <div class="metric-value">{metrics["std_area"]:.2f}</div>
                                <div class="metric-unit">standard deviation</div>
                            </div>
                            """)
                        
                        if "min_area" in metrics and "max_area" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Region Size Range</div>
                                <div class="metric-value">{metrics["min_area"]:.0f} - {metrics["max_area"]:.0f}</div>
                                <div class="metric-unit">pixels² (min-max)</div>
                            </div>
                            """)
                        
                        if "connectivity_index" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Connectivity Index</div>
                                <div class="metric-value">{metrics["connectivity_index"]:.2f}</div>
                                <div class="metric-unit">index value</div>
                            </div>
                            """)
                            
                        f.write("""
                            </div> <!-- End metrics-grid -->
                        """)
                        
                        if "connectivity" in interpretations:
                            f.write(f"""
                            <div class="interpretation">
                                <div class="interpretation-title">Connectivity Interpretation</div>
                                <p>{interpretations["connectivity"]}</p>
                            </div>
                            """)
                            
                        f.write("""
                        </div> <!-- End section -->
                        """)
                
                # Add complexity analysis section
                if "fractal" in analysis_results and "metrics" in analysis_results["fractal"]:
                    metrics = analysis_results["fractal"]["metrics"]
                    fractal_path = next((path for type, path in image_paths if type == 'fractal'), '')
                    
                    if fractal_path:
                        f.write(f"""
                        <div class="section">
                            <h2>Complexity Analysis</h2>
                            <p>
                                Complexity analysis quantifies the self-similarity and multi-scale properties
                                of the network through fractal dimension calculation. Higher fractal dimension
                                indicates more complex, space-filling structures.
                            </p>
                            
                            <div class="image-container">
                                <img src="{os.path.basename(fractal_path)}" alt="Complexity Analysis">
                                <figcaption>Binary image used for fractal dimension calculation</figcaption>
                            </div>
                            
                            <h3>Complexity Metrics</h3>
                            <div class="metrics-grid">
                        """)
                        
                        if "best_fd" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Fractal Dimension</div>
                                <div class="metric-value">{metrics["best_fd"]:.4f}</div>
                                <div class="metric-unit">D value (1.0-2.0)</div>
                            </div>
                            """)
                            
                        if "best_r2" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Goodness of Fit</div>
                                <div class="metric-value">{metrics["best_r2"]:.4f}</div>
                                <div class="metric-unit">R² value</div>
                            </div>
                            """)
                            
                        if "average_fd" in metrics:
                            f.write(f"""
                            <div class="metric-box">
                                <div class="metric-title">Average FD</div>
                                <div class="metric-value">{metrics["average_fd"]:.4f}</div>
                                <div class="metric-unit">over multiple thresholds</div>
                            </div>
                            """)
                            
                        f.write("""
                            </div> <!-- End metrics-grid -->
                        """)
                        
                        if "complexity" in interpretations:
                            f.write(f"""
                            <div class="interpretation">
                                <div class="interpretation-title">Complexity Interpretation</div>
                                <p>{interpretations["complexity"]}</p>
                                <p>
                                    Fractal dimension (FD) values range from 1.0 (simple line) to 2.0 (fills the plane completely).
                                    Filamentous biological structures typically have FD values between 1.2 and 1.8, with higher values
                                    indicating more complex, space-filling structures.
                                </p>
                            </div>
                            """)
                            
                        f.write("""
                        </div> <!-- End section -->
                        """)
                
                # Add computational models section
                if model_paths:
                    f.write(f"""
                    <div class="section">
                        <h2>Computational Morphology Models</h2>
                        <p>
                            These computational models represent different aspects of the network morphology.
                            Each model highlights specific features while abstracting away others, providing
                            insight into the network's structure, connectivity, and complexity.
                        </p>
                        
                        <div class="image-grid">
                    """)
                    
                    # Structure model
                    structure_path = next((path for type, path in model_paths if type == 'phase1'), '')
                    if structure_path:
                        f.write(f"""
                        <figure class="image-container">
                            <img src="{os.path.basename(structure_path)}" alt="Structure Model">
                            <figcaption>Structure Model - showing branching patterns and network organization</figcaption>
                        </figure>
                        """)
                    
                    # Connectivity model
                    connectivity_path = next((path for type, path in model_paths if type == 'phase2'), '')
                    if connectivity_path:
                        f.write(f"""
                        <figure class="image-container">
                            <img src="{os.path.basename(connectivity_path)}" alt="Connectivity Model">
                            <figcaption>Connectivity Model - showing spatial relationships between regions</figcaption>
                        </figure>
                        """)
                    
                    # Complexity model
                    complexity_path = next((path for type, path in model_paths if type == 'phase3'), '')
                    if complexity_path:
                        f.write(f"""
                        <figure class="image-container">
                            <img src="{os.path.basename(complexity_path)}" alt="Complexity Model">
                            <figcaption>Complexity Model - visualizing fractal patterns in the structure</figcaption>
                        </figure>
                        """)
                    
                    # Combined model
                    combined_path = next((path for type, path in model_paths if type == 'final'), '')
                    if combined_path:
                        f.write(f"""
                        <figure class="image-container" style="grid-column: 1 / -1;">
                            <img src="{os.path.basename(combined_path)}" alt="Combined Model">
                            <figcaption>Combined Morphological Model - integrating all aspects of the analysis</figcaption>
                        </figure>
                        """)
                    
                    f.write("""
                        </div> <!-- End image-grid -->
                        
                        <div class="interpretation">
                            <div class="interpretation-title">Model Interpretation</div>
                            <p>
                                These computational models extract the essential morphological characteristics of the
                                original structure while abstracting away noise and irrelevant details. They are useful
                                for comparing similar structures quantitatively and understanding how structure relates to function.
                            </p>
                        </div>
                    </div> <!-- End section -->
                    """)
                
                # Add export information and footer
                f.write(f"""
                    <div class="section">
                        <h2>Analysis Data</h2>
                        <p>
                            This report is accompanied by detailed data exports that can be used for further analysis.
                            The JSON file contains all measurements and metrics in a structured format.
                        </p>
                        
                        <p><strong>JSON Data:</strong> <a href="{os.path.basename(json_path)}" style="color: var(--accent)">{os.path.basename(json_path)}</a></p>
                        <p><strong>Generated on:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <div class="footer">
                        <p>Generated using BioMorph - Advanced Morphological Analysis</p>
                        <p>© {datetime.now().year} BioMorph - All rights reserved</p>
                    </div>
                </body>
                </html>
                """)
                
            return True, report_path
        except Exception as e:
            print(f"Error creating report: {e}")
            traceback.print_exc()
            return False, ""
    
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