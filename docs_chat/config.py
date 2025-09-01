import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.data = self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            default = {
                'llm_backend': 'gemini',  # or 'ollama'
                'gemini_api_key': 'your-gemini-api-key-here',
                'ollama_base_url': 'http://localhost:11434',
                'ollama_model': 'llama3',
                'database_path': './research_db',
                'max_context_chunks': 5
            }
            with open(self.config_path, 'w') as f:
                yaml.dump(default, f)
            return default

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        with open(self.config_path, 'w') as f:
            yaml.dump(self.data, f)
