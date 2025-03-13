import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import cv2
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import traceback

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
        
    def populate_overview_tab(self, results):
        """Populate the overview tab with key results"""
        # Create a simple placeholder
        label = self.ui.create_styled_label(
            self.overview_tab,
            "Analysis complete! Explore the tabs above to see detailed results.",
            subheader=True
        )
        label.pack(pady=50)
        
    def populate_structure_tab(self, structure_data):
        """Populate the structure analysis tab"""
        # Create a simple placeholder
        label = self.ui.create_styled_label(
            self.structure_tab,
            "Structure analysis completed. Detailed implementation coming soon.",
            subheader=True
        )
        label.pack(pady=50)
        
    def populate_connectivity_tab(self, connectivity_data):
        """Populate the connectivity analysis tab"""
        # Create a simple placeholder
        label = self.ui.create_styled_label(
            self.connectivity_tab,
            "Connectivity analysis completed. Detailed implementation coming soon.",
            subheader=True
        )
        label.pack(pady=50)
        
    def populate_complexity_tab(self, complexity_data):
        """Populate the complexity analysis tab"""
        # Create a simple placeholder
        label = self.ui.create_styled_label(
            self.complexity_tab,
            "Complexity analysis completed. Detailed implementation coming soon.",
            subheader=True
        )
        label.pack(pady=50)
        
    def populate_models_tab(self, models):
        """Populate the models tab"""
        # Create a simple placeholder
        label = self.ui.create_styled_label(
            self.models_tab,
            "Morphology models generated. Detailed implementation coming soon.",
            subheader=True
        )
        label.pack(pady=50)
        
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
    
    def export_report(self):
        """Export full analysis report"""
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
            "Generating report...",
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
                        f"Report exported to:\n{path}\n\nDo you want to open it now?"
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