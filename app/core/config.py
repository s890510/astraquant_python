import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    data_dir: str = "data"
    rss_sources: List[str] = []


class Settings(BaseSettings):
    """Application settings.

    Priority order:
    - Environment variables (.env supported)
    - config.yml values
    - Defaults
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core settings
    DATA_DIR: str = "data"
    RSS_SOURCES: List[str] = []

    # Load from config.yml if present and not overridden by env
    @classmethod
    def from_env_and_file(cls) -> "Settings":
        # Start with env/defaults
        env_settings = cls()

        # Load YAML config if exists
        cfg_path = Path("config.yml")
        file_values: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                    if isinstance(yaml_data, dict):
                        file_values = yaml_data
            except Exception:
                # ignore YAML errors, fall back to env/defaults
                file_values = {}

        # Map file values to our fields
        mapped: Dict[str, Any] = {}
        if "DATA_DIR" in file_values:
            mapped["DATA_DIR"] = str(file_values.get("DATA_DIR"))
        if "data_dir" in file_values:
            mapped["DATA_DIR"] = str(file_values.get("data_dir"))
        if "rss_sources" in file_values and isinstance(file_values["rss_sources"], list):
            mapped["RSS_SOURCES"] = [str(x) for x in file_values["rss_sources"]]
        if "RSS_SOURCES" in file_values and isinstance(file_values["RSS_SOURCES"], list):
            mapped["RSS_SOURCES"] = [str(x) for x in file_values["RSS_SOURCES"]]

        # Apply file values only where env did not set a custom value (i.e., equal to defaults)
        settings = env_settings.model_copy()
        if settings.DATA_DIR == "data" and "DATA_DIR" in mapped:
            settings.DATA_DIR = mapped["DATA_DIR"]
        if not settings.RSS_SOURCES and "RSS_SOURCES" in mapped:
            settings.RSS_SOURCES = mapped["RSS_SOURCES"]

        # Ensure data dir exists
        try:
            Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback create default data dir
            Path("data").mkdir(parents=True, exist_ok=True)
            settings.DATA_DIR = "data"

        return settings


settings = Settings.from_env_and_file()
