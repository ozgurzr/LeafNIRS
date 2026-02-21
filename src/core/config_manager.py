"""ConfigManager — User preferences and application settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class AppConfig:
    preferred_loader: str = "h5py-raw"
    recent_files: list[str] = field(default_factory=list)
    max_recent_files: int = 10
    default_visible_pairs: int = 5
    dark_theme: bool = True


class ConfigManager:
    _FILENAME = "leafnirs_config.json"

    def __init__(self, config_dir: str | None = None):
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self._path = os.path.join(config_dir, self._FILENAME)
        self.config = self._load()

    def _load(self) -> AppConfig:
        if os.path.isfile(self._path):
            try:
                with open(self._path, 'r') as f:
                    data = json.load(f)
                return AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
            except Exception:
                pass
        return AppConfig()

    def save(self):
        with open(self._path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def add_recent_file(self, filepath: str):
        if filepath in self.config.recent_files:
            self.config.recent_files.remove(filepath)
        self.config.recent_files.insert(0, filepath)
        self.config.recent_files = self.config.recent_files[:self.config.max_recent_files]
        self.save()
