import json
from pathlib import Path

class Config:
    """Configuration manager for the application"""
    DEFAULT_CONFIG = {
        "morph_open_size": 3,
        "morph_close_size": 3,
        "fractal_multi_threshold_steps": 3,
        "recent_files": [],
        "export_directory": "",
        "theme": "dark",
        "show_tooltips": True,
        "auto_run_analysis": True,
        "generate_models": True
    }
    
    def __init__(self):
        self.config_path = Path.home() / ".morphology_analyzer_config.json"
        self.data = self.load()
        
    def load(self):
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        """Get a configuration value"""
        return self.data.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value and save"""
        self.data[key] = value
        self.save()
    
    def add_recent_file(self, filepath):
        """Add a file to recent files list"""
        recent = self.data.get("recent_files", [])
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        # Keep only the 10 most recent files
        self.data["recent_files"] = recent[:10]
        self.save()