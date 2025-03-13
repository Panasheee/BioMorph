import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
import traceback

# Import our modules
from config import Config
from processor import ImageProcessor
from analysis import AnalysisPipeline
from exporter import ResultExporter
from ui import ModernUI, MorphologyAnalyzerApp

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        root.title("Morphology Analyzer")
        
        # Set initial window size and position
        root.geometry("1200x800+100+100")
        
        # Improve font rendering
        font_settings = {
            "family": "Helvetica",
            "size": 10,
            "weight": "normal"
        }
        
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(**font_settings)
        
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(**font_settings)
        
        # Initialize and run application
        app = MorphologyAnalyzerApp(root)
        
        # Set up clean exit
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the main loop
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        messagebox.showerror("Application Error", f"Error starting application: {e}")
        input("Press Enter to exit...")  # Keep console open

if __name__ == "__main__":
    main()