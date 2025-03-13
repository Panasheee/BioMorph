import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import cv2
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import traceback
from matplotlib.figure import Figure

class ModernUI:
    """Modern UI components for the application"""
    
    # Dark theme colors (base on Material Design dark theme)
    COLORS = {
        "bg_primary": "#121212",
        "bg_secondary": "#1E1E1E",
        "bg_elevated": "#2D2D2D",
        "fg_primary": "#E0E0E0", 
        "fg_secondary": "#A0A0A0",
        "accent": "#03DAC6",
        "accent_secondary": "#BB86FC",
        "warning": "#CF6679",
        "success": "#81C784"
    }
    
    def __init__(self, config):
        self.config = config
        
    def configure_styles(self):
        """Configure ttk styles for modern dark theme"""
        style = ttk.Style()
        style.theme_use("clam")  # Use a theme that supports custom styling
        
        # TFrame
        style.configure(
            "TFrame",
            background=self.COLORS["bg_primary"]
        )
        
        # Elevated.TFrame
        style.configure(
            "Elevated.TFrame",
            background=self.COLORS["bg_elevated"]
        )
        
        # Card.TFrame
        style.configure(
            "Card.TFrame",
            background=self.COLORS["bg_secondary"],
            relief="flat"
        )
        
        # TLabel
        style.configure(
            "TLabel",
            background=self.COLORS["bg_primary"],
            foreground=self.COLORS["fg_primary"],
            font=("Helvetica", 10)
        )
        
        # Title.TLabel
        style.configure(
            "Title.TLabel",
            background=self.COLORS["bg_primary"],
            foreground=self.COLORS["fg_primary"],
            font=("Helvetica", 24, "bold")
        )
        
        # Header.TLabel
        style.configure(
            "Header.TLabel",
            background=self.COLORS["bg_primary"],
            foreground=self.COLORS["fg_primary"],
            font=("Helvetica", 16, "bold")
        )
        
        # Subheader.TLabel
        style.configure(
            "Subheader.TLabel",
            background=self.COLORS["bg_primary"],
            foreground=self.COLORS["fg_primary"],
            font=("Helvetica", 12)
        )
        
        # Accent.TLabel
        style.configure(
            "Accent.TLabel",
            background=self.COLORS["bg_primary"],
            foreground=self.COLORS["accent"],
            font=("Helvetica", 10, "bold")
        )
        
        # TButton
        style.configure(
            "TButton",
            background=self.COLORS["bg_elevated"],
            foreground=self.COLORS["fg_primary"],
            borderwidth=0,
            focusthickness=0,
            font=("Helvetica", 10),
            padding=10
        )
        style.map("TButton", 
            background=[("active", self.COLORS["bg_elevated"]), 
                        ("pressed", self.COLORS["bg_elevated"])],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )
        
        # Accent.TButton
        style.configure(
            "Accent.TButton",
            background=self.COLORS["accent"],
            foreground="#000000",
            borderwidth=0,
            focusthickness=0,
            font=("Helvetica", 10, "bold"),
            padding=10
        )
        style.map("Accent.TButton", 
            background=[("active", self.COLORS["accent_secondary"]), 
                        ("pressed", self.COLORS["accent_secondary"])],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )
        
        # Large.TButton
        style.configure(
            "Large.TButton",
            background=self.COLORS["bg_elevated"],
            foreground=self.COLORS["fg_primary"],
            borderwidth=0,
            focusthickness=0,
            font=("Helvetica", 14),
            padding=15
        )
        
        # Large.Accent.TButton
        style.configure(
            "Large.Accent.TButton",
            background=self.COLORS["accent"],
            foreground="#000000",
            borderwidth=0,
            focusthickness=0,
            font=("Helvetica", 14, "bold"),
            padding=15
        )
        style.map("Large.Accent.TButton", 
            background=[("active", self.COLORS["accent_secondary"]), 
                        ("pressed", self.COLORS["accent_secondary"])],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )
        
        # TNotebook
        style.configure(
            "TNotebook",
            background=self.COLORS["bg_primary"],
            borderwidth=0,
            tabmargins=[0, 0, 0, 0]
        )
        style.configure(
            "TNotebook.Tab",
            background=self.COLORS["bg_elevated"],
            foreground=self.COLORS["fg_secondary"],
            padding=[20, 10],
            font=("Helvetica", 10)
        )
        style.map("TNotebook.Tab",
            background=[("selected", self.COLORS["accent"])],
            foreground=[("selected", "#000000")],
            expand=[("selected", [0, 0, 0, 0])]
        )
        
        # Horizontal.TProgressbar
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=self.COLORS["bg_elevated"],
            background=self.COLORS["accent"],
            thickness=10
        )
        
    def create_styled_frame(self, parent, elevated=False, card=False, **kwargs):
        """Create a styled frame"""
        if card:
            frame = ttk.Frame(parent, style="Card.TFrame", **kwargs)
        elif elevated:
            frame = ttk.Frame(parent, style="Elevated.TFrame", **kwargs)
        else:
            frame = ttk.Frame(parent, style="TFrame", **kwargs)
        return frame
        
    def create_styled_button(self, parent, text, command, accent=False, large=False, **kwargs):
        """Create a styled button"""
        if large and accent:
            style = "Large.Accent.TButton"
        elif large:
            style = "Large.TButton"
        elif accent:
            style = "Accent.TButton"
        else:
            style = "TButton"
            
        button = ttk.Button(parent, text=text, command=command, style=style, **kwargs)
        return button
        
    def create_styled_label(self, parent, text, title=False, header=False, subheader=False, accent=False, **kwargs):
        """Create a styled label"""
        if title:
            style = "Title.TLabel"
        elif header:
            style = "Header.TLabel"
        elif subheader:
            style = "Subheader.TLabel"
        elif accent:
            style = "Accent.TLabel"
        else:
            style = "TLabel"
            
        label = ttk.Label(parent, text=text, style=style, **kwargs)
        return label
        
    def create_tooltip(self, widget, text):
        """Create tooltip for a widget"""
        if not self.config.get("show_tooltips"):
            return
            
        def show_tooltip(event=None):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip, text=text, background=self.COLORS["bg_elevated"],
                foreground=self.COLORS["fg_primary"], relief="solid", borderwidth=1,
                font=("Helvetica", 10, "normal"), padx=10, pady=5
            )
            label.pack()
            def hide_tooltip(e=None):
                tooltip.destroy()
            tooltip.bind("<Leave>", hide_tooltip)
            widget.bind("<Leave>", hide_tooltip)
        widget.bind("<Enter>", show_tooltip)
        
    def create_image_preview(self, parent, width=800, height=600):
        """Create an image preview frame with a canvas for the image"""
        frame = self.create_styled_frame(parent)
        
        # Create a canvas for the image
        canvas = tk.Canvas(
            frame, 
            bg=self.COLORS["bg_primary"],
            width=width,
            height=height,
            highlightthickness=0
        )
        canvas.pack(expand=True, fill=tk.BOTH)
        
        return frame, canvas


class MorphologyAnalyzerApp:
    """Streamlined image analysis application with automatic analysis pipeline"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Morphology Analyzer")
        
        # Set initial window size
        self.root.geometry("1200x800")
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Import here to avoid circular imports
        from config import Config
        from processor import ImageProcessor
        from exporter import ResultExporter
        from analysis import AnalysisPipeline
        
        # Initialize config
        self.config = Config()
        
        # Initialize components
        self.ui = ModernUI(self.config)
        self.processor = ImageProcessor(self.config)
        self.exporter = ResultExporter(self.config)
        self.pipeline = AnalysisPipeline(self.processor, self.exporter, self.config)
        
        # Set theme and configure styles
        self.ui.configure_styles()
        
        # Set background color for root window
        self.root.configure(bg=self.ui.COLORS["bg_primary"])
        
        # Build user interface
        self.build_ui()
        
        # Track current state
        self.image_path = None
        self.is_analyzing = False
        
    def build_ui(self):
        """Build the user interface with a welcome screen and dashboard"""
        self.main_container = self.ui.create_styled_frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Initialize each view (only one is shown at a time)
        self.welcome_view = self.create_welcome_view(self.main_container)
        self.dashboard_view = self.create_dashboard_view(self.main_container)
        
        # Start with welcome view
        self.show_welcome_view()
        
    def create_welcome_view(self, parent):
        """Create the welcome screen with large upload button"""
        welcome_frame = self.ui.create_styled_frame(parent)
        
        # Center content
        content_frame = self.ui.create_styled_frame(welcome_frame)
        content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Title
        title = self.ui.create_styled_label(
            content_frame, 
            "MORPHOLOGY ANALYZER", 
            title=True
        )
        title.pack(pady=(0, 40))
        
        # Upload button
        upload_btn = self.ui.create_styled_button(
            content_frame, 
            "UPLOAD AN IMAGE", 
            self.upload_image, 
            accent=True, 
            large=True
        )
        upload_btn.pack(pady=20)
        
        # Recent files button
        if self.config.get("recent_files"):
            recent_btn = self.ui.create_styled_button(
                content_frame, 
                "RECENT FILES", 
                self.show_recent_files, 
                large=True
            )
            recent_btn.pack(pady=20)
        
        return welcome_frame
        
    def create_dashboard_view(self, parent):
        """Create the main dashboard view with notebook tabs"""
        dashboard_frame = self.ui.create_styled_frame(parent)
        
        # Top bar with controls
        top_bar = self.ui.create_styled_frame(dashboard_frame)
        top_bar.pack(fill=tk.X, pady=(10, 20))
        
        # Left side - title and back button
        title_frame = self.ui.create_styled_frame(top_bar)
        title_frame.pack(side=tk.LEFT, padx=20)
        
        back_btn = self.ui.create_styled_button(
            title_frame, 
            "← Back", 
            self.show_welcome_view
        )
        back_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        self.dashboard_title = self.ui.create_styled_label(
            title_frame, 
            "Image Analysis", 
            header=True
        )
        self.dashboard_title.pack(side=tk.LEFT)
        
        # Right side - controls
        controls_frame = self.ui.create_styled_frame(top_bar)
        controls_frame.pack(side=tk.RIGHT, padx=20)
        
        reload_btn = self.ui.create_styled_button(
            controls_frame, 
            "New Analysis", 
            self.upload_image
        )
        reload_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = self.ui.create_styled_button(
            controls_frame, 
            "Export Report", 
            self.export_report, 
            accent=True
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(dashboard_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Overview tab
        self.overview_tab = self.ui.create_styled_frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Overview")
        
        # Analysis tabs
        self.structure_tab = self.ui.create_styled_frame(self.notebook)
        self.notebook.add(self.structure_tab, text="Structure")
        
        self.connectivity_tab = self.ui.create_styled_frame(self.notebook)
        self.notebook.add(self.connectivity_tab, text="Connectivity")
        
        self.complexity_tab = self.ui.create_styled_frame(self.notebook)
        self.notebook.add(self.complexity_tab, text="Complexity")
        
        # Generated Models tab
        self.models_tab = self.ui.create_styled_frame(self.notebook)
        self.notebook.add(self.models_tab, text="Generated Models")
        
        return dashboard_frame
        
    def show_welcome_view(self):
        """Show the welcome screen"""
        self.dashboard_view.pack_forget()
        self.welcome_view.pack(fill=tk.BOTH, expand=True)
        
    def show_dashboard_view(self):
        """Show the dashboard view"""
        self.welcome_view.pack_forget()
        self.dashboard_view.pack(fill=tk.BOTH, expand=True)
        
    def update_dashboard(self, results, models):
        """Update dashboard with analysis results"""
        # Clear existing content
        for widget in self.overview_tab.winfo_children():
            widget.destroy()
        for widget in self.structure_tab.winfo_children():
            widget.destroy()
        for widget in self.connectivity_tab.winfo_children():
            widget.destroy()
        for widget in self.complexity_tab.winfo_children():
            widget.destroy()
        for widget in self.models_tab.winfo_children():
            widget.destroy()
            
        # Update title with filename
        if self.image_path:
            filename = os.path.basename(self.image_path)
            self.dashboard_title.config(text=f"Analysis: {filename}")
        
        # Populate Overview tab
        self.populate_overview_tab(results)
        
        # Populate Structure tab
        self.populate_structure_tab(results.get("medial_axis", {}))
        
        # Populate Connectivity tab
        self.populate_connectivity_tab(results.get("voronoi", {}))
        
        # Populate Complexity tab
        self.populate_complexity_tab(results.get("fractal", {}))
        
        # Populate Models tab
        self.populate_models_tab(models)
    
    def export_images(self, results):
        """Export all analysis images to files"""
        if not results:
            messagebox.showinfo("Export", "No analysis results to export.")
            return
            
        # Ask for directory to save images
        export_dir = filedialog.askdirectory(
            title="Select Directory to Save Images",
            initialdir=self.config.get("export_directory", "")
        )
        
        if not export_dir:
            return
            
        # Update export directory in config
        self.config.set("export_directory", export_dir)
        
        # Create progress dialog
        progress = tk.Toplevel(self.root)
        progress.title("Exporting Images")
        progress.configure(bg=self.ui.COLORS["bg_primary"])
        progress.geometry("300x150")
        
        # Make it modal
        progress.transient(self.root)
        progress.grab_set()
        
        # Add progress elements
        frame = self.ui.create_styled_frame(progress)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = self.ui.create_styled_label(
            frame,
            "Exporting images...",
            subheader=True
        )
        label.pack(pady=(0, 20))
        
        progress_bar = ttk.Progressbar(
            frame,
            style="Horizontal.TProgressbar",
            mode="indeterminate"
        )
        progress_bar.pack(fill=tk.X)
        progress_bar.start()
        
        # Export in a separate thread
        def export_thread():
            try:
                exported_paths = self.exporter.export_all_images(results, self.image_path)
                
                # Also export model images if available
                if hasattr(self.pipeline, "morphology_models") and self.pipeline.morphology_models:
                    model_paths = self.exporter.export_morphology_models(
                        self.pipeline.morphology_models, 
                        self.image_path
                    )
                    exported_paths.extend(model_paths)
                
                # Close progress dialog
                progress.destroy()
                
                # Show result
                if exported_paths:
                    # Format message with list of exported files
                    message = f"Successfully exported {len(exported_paths)} images:\n\n"
                    for type_name, path in exported_paths[:5]:  # Show first 5
                        message += f"• {type_name}: {os.path.basename(path)}\n"
                        
                    if len(exported_paths) > 5:
                        message += f"\n...and {len(exported_paths) - 5} more files."
                        
                    message += f"\n\nExport location: {export_dir}"
                    
                    messagebox.showinfo("Export Successful", message)
                else:
                    messagebox.showwarning("Export Notice", "No images were exported.")
            except Exception as e:
                progress.destroy()
                messagebox.showerror("Export Error", f"Error during export: {str(e)}")
                traceback.print_exc()
                
        threading.Thread(target=export_thread, daemon=True).start()

    def export_json(self, results):
        """Export analysis results to a JSON file"""
        if not results:
            messagebox.showinfo("Export", "No analysis results to export.")
            return
            
        # Ask for directory to save JSON
        export_dir = filedialog.askdirectory(
            title="Select Directory to Save JSON Data",
            initialdir=self.config.get("export_directory", "")
        )
        
        if not export_dir:
            return
            
        # Update export directory in config
        self.config.set("export_directory", export_dir)
        
        # Create progress dialog
        progress = tk.Toplevel(self.root)
        progress.title("Exporting Data")
        progress.configure(bg=self.ui.COLORS["bg_primary"])
        progress.geometry("300x150")
        
        # Make it modal
        progress.transient(self.root)
        progress.grab_set()
        
        # Add progress elements
        frame = self.ui.create_styled_frame(progress)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = self.ui.create_styled_label(
            frame,
            "Exporting data to JSON...",
            subheader=True
        )
        label.pack(pady=(0, 20))
        
        progress_bar = ttk.Progressbar(
            frame,
            style="Horizontal.TProgressbar",
            mode="indeterminate"
        )
        progress_bar.pack(fill=tk.X)
        progress_bar.start()
        
        # Export in a separate thread
        def export_thread():
            try:
                success, json_path = self.exporter.export_analysis_results(self.image_path, results)
                
                # Close progress dialog
                progress.destroy()
                
                # Show result
                if success:
                    message = f"Data exported to:\n{json_path}"
                    if messagebox.askyesno("Export Successful", f"{message}\n\nDo you want to open the containing folder?"):
                        # Open the containing folder
                        import os, subprocess
                        folder_path = os.path.dirname(json_path)
                        
                        # Platform-specific folder opening
                        if os.name == 'nt':  # Windows
                            subprocess.run(['explorer', folder_path])
                        elif os.name == 'posix':  # macOS or Linux
                            if 'darwin' in os.sys.platform:  # macOS
                                subprocess.run(['open', folder_path])
                            else:  # Linux
                                subprocess.run(['xdg-open', folder_path])
                else:
                    messagebox.showerror("Export Failed", "Failed to export data to JSON.")
            except Exception as e:
                progress.destroy()
                messagebox.showerror("Export Error", f"Error during export: {str(e)}")
                traceback.print_exc()
                
        threading.Thread(target=export_thread, daemon=True).start()

    def export_report(self):
        """Export full analysis report with enhanced styling"""
        if not hasattr(self.pipeline, "analysis_results") or not self.pipeline.analysis_results:
            messagebox.showinfo("Export", "No analysis results to export.")
            return
            
        # Create progress dialog
        progress = tk.Toplevel(self.root)
        progress.title("Exporting Report")
        progress.configure(bg=self.ui.COLORS["bg_primary"])
        progress.geometry("300x150")
        
        # Make it modal
        progress.transient(self.root)
        progress.grab_set()
        
        # Add progress bar
        frame = self.ui.create_styled_frame(progress)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = self.ui.create_styled_label(
            frame,
            "Generating comprehensive report...",
            subheader=True
        )
        label.pack(pady=(0, 20))
        
        progress_bar = ttk.Progressbar(
            frame,
            style="Horizontal.TProgressbar",
            mode="indeterminate"
        )
        progress_bar.pack(fill=tk.X)
        progress_bar.start()
        
        # Export in a separate thread
        def export_thread():
            try:
                success, path = self.exporter.export_full_report(
                    self.pipeline.analysis_results,
                    self.pipeline.morphology_models,
                    self.image_path
                )
                
                # Close progress dialog
                progress.destroy()
                
                # Show result
                if success:
                    if messagebox.askyesno(
                        "Export Successful", 
                        f"Comprehensive report exported to:\n{path}\n\nDo you want to open it now?"
                    ):
                        # Open the HTML file
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(path)}")
                else:
                    messagebox.showerror("Export Failed", "Failed to export report.")
            except Exception as e:
                progress.destroy()
                messagebox.showerror("Export Error", f"Error during export: {str(e)}")
                traceback.print_exc()
                
        threading.Thread(target=export_thread, daemon=True).start()

    def populate_overview_tab(self, results):
        """Populate the overview tab with key results and summary visualizations"""
        # Create a scrollable container
        overview_frame = self.ui.create_styled_frame(self.overview_tab)
        overview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(
            overview_frame, 
            bg=self.ui.COLORS["bg_primary"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(overview_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = self.ui.create_styled_frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        header = self.ui.create_styled_label(
            scrollable_frame,
            "Analysis Overview",
            header=True
        )
        header.pack(pady=(20, 10), padx=20)
        
        # Image summary section
        img_summary_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
        img_summary_frame.pack(fill=tk.X, expand=False, pady=10, padx=20)
        
        # Add image name and basic info
        if self.image_path:
            import os
            img_name = os.path.basename(self.image_path)
            
            file_info_frame = self.ui.create_styled_frame(img_summary_frame)
            file_info_frame.pack(fill=tk.X, padx=20, pady=20)
            
            img_label = self.ui.create_styled_label(
                file_info_frame,
                f"File: {img_name}",
                subheader=True
            )
            img_label.pack(anchor="w")
            
            # Add image dimensions
            if self.processor.original_image is not None:
                h, w = self.processor.original_image.shape[:2]
                dim_label = self.ui.create_styled_label(
                    file_info_frame,
                    f"Dimensions: {w} × {h} pixels",
                    subheader=False
                )
                dim_label.pack(anchor="w", pady=(5, 0))
        
        # Key metrics row - show the most important metrics from each analysis
        metrics_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
        metrics_frame.pack(fill=tk.X, expand=False, pady=10, padx=20)
        
        metrics_header = self.ui.create_styled_label(
            metrics_frame,
            "Key Metrics",
            subheader=True
        )
        metrics_header.pack(pady=(15, 10), padx=20)
        
        # Create a grid for metric display
        metric_grid_frame = self.ui.create_styled_frame(metrics_frame)
        metric_grid_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Add up to 3 columns of metrics
        metric_count = 0
        row = 0
        col = 0
        
        # Helper function to add a metric box
        def add_metric_box(title, value, unit="", description=""):
            nonlocal metric_count, row, col
            
            box_frame = self.ui.create_styled_frame(metric_grid_frame, elevated=True)
            box_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            title_label = self.ui.create_styled_label(box_frame, title, accent=True)
            title_label.pack(pady=(10, 5), padx=10)
            
            value_str = f"{value}{' ' + unit if unit else ''}"
            # Fix the syntax error with the conditional styling
            is_short = len(value_str) < 10
            value_label = self.ui.create_styled_label(
                box_frame, value_str, 
                header=is_short,
                subheader=not is_short
            )
            value_label.pack(pady=5, padx=10)
            
            if description:
                desc_label = self.ui.create_styled_label(box_frame, description)
                desc_label.pack(pady=(5, 10), padx=10)
            
            # Update position
            metric_count += 1
            col += 1
            if col > 2:  # 3 columns (0, 1, 2)
                col = 0
                row += 1
        
        # Add structure metrics
        if "medial_axis" in results and results["medial_axis"].get("metrics"):
            metrics = results["medial_axis"]["metrics"]
            if "branch_points" in metrics:
                add_metric_box(
                    "Branch Points", 
                    metrics["branch_points"], 
                    description="Junctions in the network"
                )
            if "branches_per_area" in metrics:
                add_metric_box(
                    "Branch Density", 
                    f"{metrics['branches_per_area']:.2f}", 
                    "per 1000px²",
                    "Network complexity"
                )
        
        # Add connectivity metrics
        if "voronoi" in results and results["voronoi"].get("metrics"):
            metrics = results["voronoi"]["metrics"]
            if "num_regions" in metrics:
                add_metric_box(
                    "Network Regions", 
                    metrics["num_regions"], 
                    description="Partitioned areas"
                )
        
        # Add fractal metrics
        if "fractal" in results and results["fractal"].get("metrics"):
            metrics = results["fractal"]["metrics"]
            if "best_fd" in metrics:
                add_metric_box(
                    "Fractal Dimension", 
                    f"{metrics['best_fd']:.3f}", 
                    description="Structural complexity"
                )
        
        # Configure grid to expand properly
        for i in range(3):
            metric_grid_frame.columnconfigure(i, weight=1)
        
        # Analysis thumbnails section
        thumbnails_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
        thumbnails_frame.pack(fill=tk.X, expand=False, pady=10, padx=20)
        
        thumbnails_header = self.ui.create_styled_label(
            thumbnails_frame,
            "Analysis Visualizations",
            subheader=True
        )
        thumbnails_header.pack(pady=(15, 10), padx=20)
        
        # Create a grid for thumbnails
        thumb_grid_frame = self.ui.create_styled_frame(thumbnails_frame)
        thumb_grid_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Add thumbnails for each analysis
        thumb_size = (200, 150)  # width, height
        row = 0
        col = 0
        
        # Helper function to add a thumbnail
        def add_thumbnail(image, title, tab_index):
            nonlocal row, col
            
            thumb_frame = self.ui.create_styled_frame(thumb_grid_frame, elevated=True)
            thumb_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Create the thumbnail
            if image is not None:
                # Resize image
                h, w = image.shape[:2]
                aspect = w / h
                
                if aspect > thumb_size[0] / thumb_size[1]:  # wider than tall
                    new_w = thumb_size[0]
                    new_h = int(new_w / aspect)
                else:  # taller than wide
                    new_h = thumb_size[1]
                    new_w = int(new_h * aspect)
                    
                thumb = cv2.resize(image, (new_w, new_h))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
                
                # Convert to PhotoImage
                pil_img = Image.fromarray(thumb)
                tk_img = ImageTk.PhotoImage(pil_img)
                
                # Keep a reference to the image to prevent garbage collection
                thumb_frame.image = tk_img
                
                # Add image label
                img_label = tk.Label(
                    thumb_frame, 
                    image=tk_img,
                    bg=self.ui.COLORS["bg_elevated"],
                    bd=0
                )
                img_label.pack(pady=5)
                
                # Make it clickable to go to corresponding tab
                img_label.bind("<Button-1>", lambda e: self.notebook.select(tab_index))
                
            # Add title
            title_label = self.ui.create_styled_label(thumb_frame, title, accent=True)
            title_label.pack(pady=5)
            
            # Make title clickable to go to corresponding tab
            title_label.bind("<Button-1>", lambda e: self.notebook.select(tab_index))
            
            # Update position
            col += 1
            if col > 2:  # 3 columns (0, 1, 2)
                col = 0
                row += 1
        
        # Add thumbnails for each analysis
        if "binary" in results and results["binary"].get("image") is not None:
            add_thumbnail(results["binary"]["image"], "Binary Segmentation", 0)
            
        if "medial_axis" in results and results["medial_axis"].get("image") is not None:
            add_thumbnail(results["medial_axis"]["image"], "Structure Analysis", 1)
            
        if "voronoi" in results and results["voronoi"].get("image") is not None:
            add_thumbnail(results["voronoi"]["image"], "Connectivity Analysis", 2)
            
        if "fractal" in results and results["fractal"].get("image") is not None:
            add_thumbnail(results["fractal"]["image"], "Complexity Analysis", 3)
            
        # Configure grid to expand properly
        for i in range(3):
            thumb_grid_frame.columnconfigure(i, weight=1)
        
        # Export options section
        export_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
        export_frame.pack(fill=tk.X, expand=False, pady=(10, 20), padx=20)
        
        export_title = self.ui.create_styled_label(
            export_frame,
            "Export Options",
            subheader=True
        )
        export_title.pack(pady=(15, 10), padx=20)
        
        button_frame = self.ui.create_styled_frame(export_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Add export buttons
        export_report_btn = self.ui.create_styled_button(
            button_frame,
            "Export Full Report",
            self.export_report,
            accent=True
        )
        export_report_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        export_images_btn = self.ui.create_styled_button(
            button_frame,
            "Export Images",
            lambda: self.export_images(results)
        )
        export_images_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        export_data_btn = self.ui.create_styled_button(
            button_frame,
            "Export Data (JSON)",
            lambda: self.export_json(results)
        )
        export_data_btn.pack(side=tk.LEFT, padx=5, pady=5)

    def populate_structure_tab(self, structure_data):
        """Populate the structure tab with medial axis/skeleton analysis"""
        # Check if we have valid data
        if not structure_data or "image" not in structure_data or structure_data["image"] is None:
            label = self.ui.create_styled_label(
                self.structure_tab,
                "No valid structural analysis data available.",
                subheader=True
            )
            label.pack(pady=50)
            return
        
        # Create main container
        container = self.ui.create_styled_frame(self.structure_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Split into left (image) and right (info) panels
        left_panel = self.ui.create_styled_frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = self.ui.create_styled_frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(20, 0))
        
        # Add title to right panel
        title = self.ui.create_styled_label(
            right_panel,
            "Structure Analysis",
            header=True
        )
        title.pack(pady=(0, 20))
        
        # Display the image in left panel
        image = structure_data["image"]
        h, w = image.shape[:2]
        
        # Calculate max dimensions to fit in panel
        max_w = 800
        max_h = 600
        
        # Resize if needed
        if w > max_w or h > max_h:
            aspect = w / h
            if aspect > max_w / max_h:  # wider than tall
                new_w = max_w
                new_h = int(new_w / aspect)
            else:  # taller than wide
                new_h = max_h
                new_w = int(new_h * aspect)
                
            display_img = cv2.resize(image, (new_w, new_h))
        else:
            display_img = image
        
        # Convert to PhotoImage
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Create image container
        img_frame = self.ui.create_styled_frame(left_panel, card=True)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Keep a reference to the image
        img_frame.image = tk_img
        
        # Add image
        img_label = tk.Label(
            img_frame,
            image=tk_img,
            bg=self.ui.COLORS["bg_secondary"],
            bd=0
        )
        img_label.pack(padx=10, pady=10)
        
        # Add info panels on right side
        # Add metrics card
        metrics_frame = self.ui.create_styled_frame(right_panel, card=True)
        metrics_frame.pack(fill=tk.X, expand=False, pady=(0, 20))
        
        metrics_title = self.ui.create_styled_label(
            metrics_frame,
            "Network Metrics",
            subheader=True
        )
        metrics_title.pack(pady=(15, 10), padx=20)
        
        # Add metrics from the data
        metrics = structure_data.get("metrics", {})
        
        metrics_grid = self.ui.create_styled_frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, expand=False, padx=20, pady=(0, 15))
        
        # Add metrics in a grid layout
        row = 0
        
        def add_metric_row(label_text, value, unit=""):
            nonlocal row
            
            label = self.ui.create_styled_label(metrics_grid, label_text)
            label.grid(row=row, column=0, sticky="w", pady=2)
            
            value_str = f"{value}"
            if unit:
                value_str += f" {unit}"
                
            value_label = self.ui.create_styled_label(metrics_grid, value_str, accent=True)
            value_label.grid(row=row, column=1, sticky="e", pady=2)
            
            row += 1
        
        # Add all available metrics
        if "branch_points" in metrics:
            add_metric_row("Branch Points:", metrics["branch_points"])
            
        if "end_points" in metrics:
            add_metric_row("End Points:", metrics["end_points"])
            
        if "skeleton_pixels" in metrics:
            add_metric_row("Skeleton Length:", metrics["skeleton_pixels"], "pixels")
            
        if "branches_per_area" in metrics:
            add_metric_row("Branch Density:", f"{metrics['branches_per_area']:.2f}", "per 1000px²")
            
        if "total_branches" in metrics:
            add_metric_row("Total Branches:", metrics["total_branches"])
            
        if "avg_branch_length" in metrics:
            add_metric_row("Avg. Branch Length:", f"{metrics['avg_branch_length']:.2f}", "pixels")
            
        if "endpoint_ratio" in metrics:
            add_metric_row("Endpoint/Branch Ratio:", f"{metrics['endpoint_ratio']:.2f}")
            
        if "network_complexity" in metrics:
            add_metric_row("Network Complexity:", f"{metrics['network_complexity']:.2f}")
        
        # Configure grid columns
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
        # Add explanation of color coding
        legend_frame = self.ui.create_styled_frame(right_panel, card=True)
        legend_frame.pack(fill=tk.X, expand=False)
        
        legend_title = self.ui.create_styled_label(
            legend_frame,
            "Color Legend",
            subheader=True
        )
        legend_title.pack(pady=(15, 10), padx=20)
        
        legend_items = [
            ("Yellow", "Branch Points (Network Junctions)"),
            ("Magenta", "End Points (Network Terminals)"),
            ("Red", "Main Skeleton (Network Pathways)")
        ]
        
        legend_grid = self.ui.create_styled_frame(legend_frame)
        legend_grid.pack(fill=tk.X, expand=False, padx=20, pady=(0, 15))
        
        for i, (color, desc) in enumerate(legend_items):
            # Create color box
            color_box = tk.Canvas(
                legend_grid, 
                width=15, 
                height=15, 
                bg=color,
                bd=0, 
                highlightthickness=0
            )
            color_box.grid(row=i, column=0, pady=5, padx=5, sticky="w")
            
            # Create description
            desc_label = self.ui.create_styled_label(legend_grid, desc)
            desc_label.grid(row=i, column=1, pady=5, padx=5, sticky="w")
        
        # Add interpretation panel
        interp_frame = self.ui.create_styled_frame(right_panel, card=True)
        interp_frame.pack(fill=tk.X, expand=False, pady=(20, 0))
        
        interp_title = self.ui.create_styled_label(
            interp_frame,
            "Interpretation",
            subheader=True
        )
        interp_title.pack(pady=(15, 10), padx=20)
        
        # Generate an interpretation based on metrics
        if metrics:
            branch_density = metrics.get("branches_per_area", 0)
            endpoint_ratio = metrics.get("endpoint_ratio", 0)
            
            interpretation = "Network Characteristics:\n\n"
            
            if branch_density < 0.5:
                interpretation += "• Low branching density indicates a simple network with minimal complexity.\n"
            elif branch_density < 2.0:
                interpretation += "• Moderate branching density suggests a balanced network structure.\n"
            else:
                interpretation += "• High branching density indicates a complex, highly interconnected network.\n"
                
            if endpoint_ratio < 1.0:
                interpretation += "• Low endpoint ratio suggests a closed or circular network structure.\n"
            elif endpoint_ratio < 2.0:
                interpretation += "• Balanced endpoint ratio indicates a mix of closed and open pathways.\n"
            else:
                interpretation += "• High endpoint ratio suggests an open network with many terminating branches.\n"
        else:
            interpretation = "No metrics available for interpretation."
        
        interp_text = self.ui.create_styled_label(interp_frame, interpretation)
        interp_text.pack(pady=(0, 15), padx=20, anchor="w")

    def populate_connectivity_tab(self, connectivity_data):
        """Populate the connectivity tab with Voronoi analysis"""
        # Check if we have valid data
        if not connectivity_data or "image" not in connectivity_data or connectivity_data["image"] is None:
            label = self.ui.create_styled_label(
                self.connectivity_tab,
                "No valid connectivity analysis data available.",
                subheader=True
            )
            label.pack(pady=50)
            return
        
        # Create main container
        container = self.ui.create_styled_frame(self.connectivity_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Split into left (image) and right (info) panels
        left_panel = self.ui.create_styled_frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = self.ui.create_styled_frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(20, 0))
        
        # Add title to right panel
        title = self.ui.create_styled_label(
            right_panel,
            "Connectivity Analysis",
            header=True
        )
        title.pack(pady=(0, 20))
        
        # Display the image in left panel
        image = connectivity_data["image"]
        h, w = image.shape[:2]
        
        # Calculate max dimensions to fit in panel
        max_w = 800
        max_h = 600
        
        # Resize if needed
        if w > max_w or h > max_h:
            aspect = w / h
            if aspect > max_w / max_h:  # wider than tall
                new_w = max_w
                new_h = int(new_w / aspect)
            else:  # taller than wide
                new_h = max_h
                new_w = int(new_h * aspect)
                
            display_img = cv2.resize(image, (new_w, new_h))
        else:
            display_img = image
        
        # Convert to PhotoImage
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Create image container
        img_frame = self.ui.create_styled_frame(left_panel, card=True)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Keep a reference to the image
        img_frame.image = tk_img
        
        # Add image
        img_label = tk.Label(
            img_frame,
            image=tk_img,
            bg=self.ui.COLORS["bg_secondary"],
            bd=0
        )
        img_label.pack(padx=10, pady=10)
        
        # Add info panels on right side
        # Add metrics card
        metrics_frame = self.ui.create_styled_frame(right_panel, card=True)
        metrics_frame.pack(fill=tk.X, expand=False, pady=(0, 20))
        
        metrics_title = self.ui.create_styled_label(
            metrics_frame,
            "Region Metrics",
            subheader=True
        )
        metrics_title.pack(pady=(15, 10), padx=20)
        
        # Add metrics from the data
        metrics = connectivity_data.get("metrics", {})
        
        metrics_grid = self.ui.create_styled_frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, expand=False, padx=20, pady=(0, 15))
        
        # Add metrics in a grid layout
        row = 0
        
        def add_metric_row(label_text, value, unit=""):
            nonlocal row
            
            label = self.ui.create_styled_label(metrics_grid, label_text)
            label.grid(row=row, column=0, sticky="w", pady=2)
            
            value_str = f"{value}"
            if unit:
                value_str += f" {unit}"
                
            value_label = self.ui.create_styled_label(metrics_grid, value_str, accent=True)
            value_label.grid(row=row, column=1, sticky="e", pady=2)
            
            row += 1
        
        # Add all available metrics
        if "num_regions" in metrics:
            add_metric_row("Number of Regions:", metrics["num_regions"])
            
        if "mean_area" in metrics:
            add_metric_row("Mean Region Area:", f"{metrics['mean_area']:.2f}", "pixels²")
            
        if "std_area" in metrics:
            add_metric_row("Area Standard Deviation:", f"{metrics['std_area']:.2f}", "pixels²")
            
        if "min_area" in metrics:
            add_metric_row("Minimum Region Area:", f"{metrics['min_area']:.2f}", "pixels²")
            
        if "max_area" in metrics:
            add_metric_row("Maximum Region Area:", f"{metrics['max_area']:.2f}", "pixels²")
            
        if "connectivity_index" in metrics:
            add_metric_row("Connectivity Index:", f"{metrics['connectivity_index']:.2f}")
        
        # Configure grid columns
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
        # Add explanation of Voronoi analysis
        explanation_frame = self.ui.create_styled_frame(right_panel, card=True)
        explanation_frame.pack(fill=tk.X, expand=False)
        
        explanation_title = self.ui.create_styled_label(
            explanation_frame,
            "About Voronoi Analysis",
            subheader=True
        )
        explanation_title.pack(pady=(15, 10), padx=20)
        
        explanation_text = """
Voronoi analysis divides the image into regions based on network branching points and structure centers.

Each colored region represents an area closest to a specific structural element.

This analysis helps understand spatial distribution, connectivity, and regional influence of different parts of the network.

Smaller, more uniform regions indicate more evenly distributed network elements.
        """
        
        explanation_label = self.ui.create_styled_label(explanation_frame, explanation_text.strip())
        explanation_label.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Add interpretation panel
        interp_frame = self.ui.create_styled_frame(right_panel, card=True)
        interp_frame.pack(fill=tk.X, expand=False, pady=(20, 0))
        
        interp_title = self.ui.create_styled_label(
            interp_frame,
            "Interpretation",
            subheader=True
        )
        interp_title.pack(pady=(15, 10), padx=20)
        
        # Generate an interpretation based on metrics
        if metrics:
            num_regions = metrics.get("num_regions", 0)
            mean_area = metrics.get("mean_area", 0)
            std_area = metrics.get("std_area", 0)
            
            cv = std_area / mean_area if mean_area > 0 else 0  # Coefficient of variation
            
            interpretation = "Region Analysis:\n\n"
            
            # Interpret number of regions
            if num_regions < 10:
                interpretation += "• Low number of regions suggests a sparse network with limited connectivity.\n"
            elif num_regions < 50:
                interpretation += "• Moderate number of regions indicates a well-developed network structure.\n"
            else:
                interpretation += "• High number of regions suggests a dense, highly connected network.\n"
                
            # Interpret area variation
            if cv < 0.5:
                interpretation += "• Low variation in region sizes indicates uniform network distribution.\n"
            elif cv < 1.0:
                interpretation += "• Moderate variation in region sizes suggests a balanced network with some specialized areas.\n"
            else:
                interpretation += "• High variation in region sizes indicates heterogeneous network with some dominant structural elements.\n"
        else:
            interpretation = "No metrics available for interpretation."
        
        interp_text = self.ui.create_styled_label(interp_frame, interpretation)
        interp_text.pack(pady=(0, 15), padx=20, anchor="w")

    def populate_complexity_tab(self, complexity_data):
        """Populate the complexity tab with fractal analysis results"""
        # Check if we have valid data
        if not complexity_data or "image" not in complexity_data or complexity_data["image"] is None:
            label = self.ui.create_styled_label(
                self.complexity_tab,
                "No valid complexity analysis data available.",
                subheader=True
            )
            label.pack(pady=50)
            return
        
        # Create main container
        container = self.ui.create_styled_frame(self.complexity_tab)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Split into left (image) and right (info) panels
        left_panel = self.ui.create_styled_frame(container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = self.ui.create_styled_frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(20, 0))
        
        # Add title to right panel
        title = self.ui.create_styled_label(
            right_panel,
            "Complexity Analysis",
            header=True
        )
        title.pack(pady=(0, 20))
        
        # Display the image in left panel
        image = complexity_data["image"]
        h, w = image.shape[:2]
        
        # Calculate max dimensions to fit in panel
        max_w = 800
        max_h = 600
        
        # Resize if needed
        if w > max_w or h > max_h:
            aspect = w / h
            if aspect > max_w / max_h:  # wider than tall
                new_w = max_w
                new_h = int(new_w / aspect)
            else:  # taller than wide
                new_h = max_h
                new_w = int(new_h * aspect)
                
            display_img = cv2.resize(image, (new_w, new_h))
        else:
            display_img = image
        
        # Convert to PhotoImage
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Create image container
        img_frame = self.ui.create_styled_frame(left_panel, card=True)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Keep a reference to the image
        img_frame.image = tk_img
        
        # Add image
        img_label = tk.Label(
            img_frame,
            image=tk_img,
            bg=self.ui.COLORS["bg_secondary"],
            bd=0
        )
        img_label.pack(padx=10, pady=10)
        
        # Add fractal dimension metrics card
        metrics_frame = self.ui.create_styled_frame(right_panel, card=True)
        metrics_frame.pack(fill=tk.X, expand=False, pady=(0, 20))
        
        metrics_title = self.ui.create_styled_label(
            metrics_frame,
            "Fractal Dimension",
            subheader=True
        )
        metrics_title.pack(pady=(15, 10), padx=20)
        
        # Add metrics from the data
        metrics = complexity_data.get("metrics", {})
        
        metrics_grid = self.ui.create_styled_frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, expand=False, padx=20, pady=(0, 15))
        
        # Add metrics in a grid layout
        row = 0
        
        def add_metric_row(label_text, value, unit=""):
            nonlocal row
            
            label = self.ui.create_styled_label(metrics_grid, label_text)
            label.grid(row=row, column=0, sticky="w", pady=2)
            
            value_str = f"{value}"
            if unit:
                value_str += f" {unit}"
                
            value_label = self.ui.create_styled_label(metrics_grid, value_str, accent=True)
            value_label.grid(row=row, column=1, sticky="e", pady=2)
            
            row += 1
        
        # Add all available metrics
        if "best_fd" in metrics:
            add_metric_row("Best Fractal Dimension:", f"{metrics['best_fd']:.4f}")
            
        if "best_r2" in metrics:
            add_metric_row("R² Value:", f"{metrics['best_r2']:.4f}")
            
        if "average_fd" in metrics:
            add_metric_row("Average Fractal Dimension:", f"{metrics['average_fd']:.4f}")
        
        # Configure grid columns
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
        # Add a plot of box counting if available
        if complexity_data.get("details") and len(complexity_data["details"]) > 0:
            # Find the best one by R²
            best_detail = max(complexity_data["details"], key=lambda x: x.get("r_squared", 0))
            
            if "log_sizes" in best_detail and "log_counts" in best_detail:
                log_sizes = best_detail["log_sizes"]
                log_counts = best_detail["log_counts"]
                
                if len(log_sizes) > 0 and len(log_counts) > 0:
                    # Create a plot
                    plot_frame = self.ui.create_styled_frame(right_panel, card=True)
                    plot_frame.pack(fill=tk.X, expand=False, pady=(0, 20))
                    
                    plot_title = self.ui.create_styled_label(
                        plot_frame,
                        "Box Counting Plot",
                        subheader=True
                    )
                    plot_title.pack(pady=(15, 10), padx=20)
                    
                    # Create matplotlib figure
                    fig = Figure(figsize=(4, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    
                    # Plot data
                    ax.scatter(log_sizes, log_counts, color='#03DAC6')
                    
                    # Add regression line
                    import numpy as np
                    coeffs = np.polyfit(log_sizes, log_counts, 1)
                    slope, intercept = coeffs
                    x_range = np.array([min(log_sizes), max(log_sizes)])
                    y_range = slope * x_range + intercept
                    ax.plot(x_range, y_range, color='#BB86FC', linestyle='--')
                    
                    # Add annotations
                    fd = -slope
                    ax.set_title(f'Fractal Dimension: {fd:.4f}')
                    ax.set_xlabel('log(Box Size)')
                    ax.set_ylabel('log(Count)')
                    
                    # Set background color
                    fig.patch.set_facecolor(self.ui.COLORS["bg_secondary"])
                    ax.set_facecolor(self.ui.COLORS["bg_secondary"])
                    
                    # Set text color to white
                    for text in ax.get_xticklabels() + ax.get_yticklabels():
                        text.set_color(self.ui.COLORS["fg_primary"])
                    ax.title.set_color(self.ui.COLORS["fg_primary"])
                    ax.xaxis.label.set_color(self.ui.COLORS["fg_primary"])
                    ax.yaxis.label.set_color(self.ui.COLORS["fg_primary"])
                    
                    # Set grid and spines color
                    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                    for spine in ax.spines.values():
                        spine.set_color('gray')
                    
                    # Add the equation
                    eq_text = f'y = {slope:.2f}x + {intercept:.2f}'
                    ax.annotate(eq_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                               color=self.ui.COLORS["fg_primary"])
                    
                    # Create canvas and add to frame
                    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(pady=(0, 15), padx=20)
        
        # Add explanation of fractal dimension
        explanation_frame = self.ui.create_styled_frame(right_panel, card=True)
        explanation_frame.pack(fill=tk.X, expand=False)
        
        explanation_title = self.ui.create_styled_label(
            explanation_frame,
            "About Fractal Dimension",
            subheader=True
        )
        explanation_title.pack(pady=(15, 10), padx=20)
        
        explanation_text = """
Fractal dimension (FD) quantifies the complexity of a pattern by measuring how detail changes with scale.

• FD values range from 1.0 (simple line) to 2.0 (fills the plane completely)
• Higher values indicate more complex, space-filling structures
• Biological structures typically have FD between 1.2 and 1.8

The R² value indicates how well the data fits the fractal model (closer to 1.0 is better).
        """
        
        explanation_label = self.ui.create_styled_label(explanation_frame, explanation_text.strip())
        explanation_label.pack(pady=(0, 15), padx=20, anchor="w")
        
        # Add interpretation panel
        interp_frame = self.ui.create_styled_frame(right_panel, card=True)
        interp_frame.pack(fill=tk.X, expand=False, pady=(20, 0))
        
        interp_title = self.ui.create_styled_label(
            interp_frame,
            "Interpretation",
            subheader=True
        )
        interp_title.pack(pady=(15, 10), padx=20)
        
        # Generate an interpretation based on metrics
        interpretation = "No metrics available for interpretation."
        
        if "fractal_dimension_interpretation" in metrics:
            interpretation = metrics["fractal_dimension_interpretation"]
        elif "best_fd" in metrics:
            fd = metrics["best_fd"]
            r2 = metrics.get("best_r2", 0)
            
            interpretation = "Structure Complexity:\n\n"
            
            # Interpret fractal dimension
            if fd < 1.2:
                interpretation += "• Low fractal dimension (< 1.2) indicates a simple structure with minimal branching.\n"
            elif fd < 1.4:
                interpretation += "• Moderate fractal dimension (1.2 - 1.4) suggests a structure with some branching and complexity.\n"
            elif fd < 1.6:
                interpretation += "• High fractal dimension (1.4 - 1.6) indicates a complex network with significant branching.\n"
            elif fd < 1.8:
                interpretation += "• Very high fractal dimension (1.6 - 1.8) suggests a highly complex, space-filling structure.\n"
            else:
                interpretation += "• Extremely high fractal dimension (> 1.8) indicates an exceptionally complex structure approaching plane-filling.\n"
                
            # Interpret goodness of fit
            if r2 < 0.9:
                interpretation += "\n• The lower R² value suggests that this structure may not follow strictly fractal scaling laws.\n"
            else:
                interpretation += "\n• The high R² value confirms strong fractal properties in this structure.\n"
        
        interp_text = self.ui.create_styled_label(interp_frame, interpretation)
        interp_text.pack(pady=(0, 15), padx=20, anchor="w")

    def populate_models_tab(self, models):
        """Populate the models tab with generated computational models"""
        # Check if we have valid models
        if not models or not any(models.values()):
            label = self.ui.create_styled_label(
                self.models_tab,
                "No valid morphology models available.",
                subheader=True
            )
            label.pack(pady=50)
            return
        
        # Create scrollable container
        models_frame = self.ui.create_styled_frame(self.models_tab)
        models_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(
            models_frame, 
            bg=self.ui.COLORS["bg_primary"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(models_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = self.ui.create_styled_frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        header = self.ui.create_styled_label(
            scrollable_frame,
            "Computational Morphology Models",
            header=True
        )
        header.pack(pady=(20, 10), padx=20)
        
        # Description
        description = self.ui.create_styled_label(
            scrollable_frame,
            "These models are computational representations of the analyzed structure, highlighting different aspects of morphology.",
            subheader=True
        )
        description.pack(pady=(0, 20), padx=20)
        
        # Helper function to add a model display
        def add_model_display(image, title, description):
            if image is None:
                return
                
            # Create frame for this model
            model_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
            model_frame.pack(fill=tk.X, expand=False, pady=10, padx=20)
            
            # Add title
            model_title = self.ui.create_styled_label(
                model_frame,
                title,
                subheader=True
            )
            model_title.pack(pady=(15, 10), padx=20)
            
            # Create two columns: image on left, description on right
            columns_frame = self.ui.create_styled_frame(model_frame)
            columns_frame.pack(fill=tk.X, expand=False, padx=20, pady=(0, 15))
            
            # Left column for image
            left_col = self.ui.create_styled_frame(columns_frame)
            left_col.pack(side=tk.LEFT, fill=tk.Y, expand=False)
            
            # Right column for description
            right_col = self.ui.create_styled_frame(columns_frame)
            right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
            
            # Resize image
            h, w = image.shape[:2]
            max_size = 300
            
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
                
            resized = cv2.resize(image, (new_w, new_h))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            pil_img = Image.fromarray(resized)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Keep a reference to the image
            left_col.image = tk_img
            
            # Add image
            img_label = tk.Label(
                left_col,
                image=tk_img,
                bg=self.ui.COLORS["bg_secondary"],
                bd=0
            )
            img_label.pack(padx=10, pady=10)
            
            # Add description
            desc_label = self.ui.create_styled_label(right_col, description)
            desc_label.pack(pady=10, anchor="w")
        
        # Add structure model
        if "phase1" in models and models["phase1"] is not None:
            add_model_display(
                models["phase1"],
                "Structure Model",
                "This model emphasizes the branching patterns and structural organization of the network. "
                "Yellow nodes represent branch points, while magenta points show endpoints. "
                "The overall branching pattern is derived from the medial axis analysis."
            )
        
        # Add connectivity model
        if "phase2" in models and models["phase2"] is not None:
            add_model_display(
                models["phase2"],
                "Connectivity Model",
                "This model represents the spatial organization and connectivity between different regions. "
                "Color-coded areas show the Voronoi regions that divide the space based on structural elements. "
                "The size and distribution of these regions indicate how different parts of the network interact spatially."
            )
        
        # Add complexity model
        if "phase3" in models and models["phase3"] is not None:
            add_model_display(
                models["phase3"],
                "Complexity Model",
                "This model visualizes the fractal complexity of the structure. "
                "The recursive patterns demonstrate the self-similarity at different scales observed in the original image. "
                "More intricate and densely packed patterns indicate higher fractal dimension and complexity."
            )
        
        # Add final combined model
        if "final" in models and models["final"] is not None:
            add_model_display(
                models["final"],
                "Combined Morphological Model",
                "This integrated model combines aspects from all three analyses: structure, connectivity, and complexity. "
                "It provides a comprehensive representation of the morphological characteristics, highlighting how "
                "branching patterns, spatial organization, and complexity interact to form the complete structure."
            )
        
        # Add explanation
        explanation_frame = self.ui.create_styled_frame(scrollable_frame, card=True)
        explanation_frame.pack(fill=tk.X, expand=False, pady=(10, 20), padx=20)
        
        explanation_title = self.ui.create_styled_label(
            explanation_frame,
            "About Computational Morphology Models",
            subheader=True
        )
        explanation_title.pack(pady=(15, 10), padx=20)
        
        explanation_text = """
Computational morphology models are algorithmic representations of the original structure, 
highlighting specific aspects while abstracting away noise and irrelevant details.

These models are useful for:

• Comparing similar structures quantitatively
• Identifying key morphological patterns
• Understanding how structure relates to function
• Creating simulations based on real-world morphology

The models maintain the essential characteristics of the original structure while providing
a simplified representation that emphasizes specific properties of interest.
        """
        
        explanation_label = self.ui.create_styled_label(explanation_frame, explanation_text.strip())
        explanation_label.pack(pady=(0, 15), padx=20, anchor="w")

    def upload_image(self):
        """Open an image file dialog"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            # Set dashboard title to show loading state
            self.show_dashboard_view()
            self.dashboard_title.config(text=f"Loading: {os.path.basename(file_path)}...")
            
            # Use threading to keep UI responsive
            threading.Thread(target=self.run_analysis, args=(file_path,), daemon=True).start()
        
    def run_analysis(self, image_path):
        """Run full analysis pipeline on an image"""
        try:
            # Mark as analyzing
            self.is_analyzing = True
            self.image_path = image_path
            
            # Add to recent files
            self.config.add_recent_file(image_path)
            
            # Run the analysis pipeline
            success = self.pipeline.run_full_analysis(image_path)
            
            # Update UI on main thread
            if success:
                self.root.after(0, lambda: self.update_dashboard(
                    self.pipeline.analysis_results,
                    self.pipeline.morphology_models
                ))
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Analysis Failed", 
                    "Failed to analyze image."
                ))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", 
                f"Error during analysis: {str(e)}"
            ))
            traceback.print_exc()
        finally:
            self.is_analyzing = False
        
    def show_recent_files(self):
        """Show recent files dialog"""
        recent_files = self.config.get("recent_files", [])
        if not recent_files:
            messagebox.showinfo("Recent Files", "No recent files.")
            return
            
        # Create a simple dialog with list of files
        dialog = tk.Toplevel(self.root)
        dialog.title("Recent Files")
        dialog.configure(bg=self.ui.COLORS["bg_primary"])
        dialog.geometry("500x400")
        
        # Make it modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create a styled frame
        frame = self.ui.create_styled_frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = self.ui.create_styled_label(
            frame, 
            "Recent Files", 
            header=True
        )
        title.pack(pady=(0, 20))
        
        # Create a list frame
        list_frame = self.ui.create_styled_frame(frame, card=True)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add files to list
        for filepath in recent_files:
            if os.path.exists(filepath):
                filename = os.path.basename(filepath)
                file_btn = self.ui.create_styled_button(
                    list_frame,
                    filename,
                    lambda f=filepath: self.open_recent_file(f, dialog)
                )
                file_btn.pack(fill=tk.X, padx=10, pady=2)
        
        # Cancel button
        cancel_btn = self.ui.create_styled_button(
            frame,
            "Cancel",
            dialog.destroy
        )
        cancel_btn.pack(pady=(20, 0))
    
    def open_recent_file(self, filepath, dialog=None):
        """Open a recent file and start analysis"""
        if dialog:
            dialog.destroy()
            
        # Show dashboard and start analysis
        self.show_dashboard_view()
        self.dashboard_title.config(text=f"Loading: {os.path.basename(filepath)}...")
        
        # Start analysis thread
        threading.Thread(target=self.run_analysis, args=(filepath,), daemon=True).start()