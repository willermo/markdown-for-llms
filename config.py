#!/usr/bin/env python3
"""
Centralized configuration management for the Document Intelligence Pipeline
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any, List
from enum import Enum

class LLMModel(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    CLAUDE_3 = "claude-3"
    LLAMA_2 = "llama-2"
    GEMINI_PRO = "gemini-pro"
    CUSTOM = "custom"

@dataclass
class DirectoryConfig:
    """Directory structure configuration"""
    source_pdfs: str = "source_pdfs"
    source_documents: str = "source_documents"
    converted: str = "converted_markdown"
    cleaned: str = "cleaned_markdown"
    validated: str = "validated_markdown"
    chunked: str = "chunked_markdown"
    logs: str = "pipeline_logs"

@dataclass
class LLMPreset:
    """LLM-specific chunking configuration"""
    chunk_size: int
    overlap: int
    description: str = ""

@dataclass
class ValidationThresholds:
    """Validation quality thresholds"""
    min_content_length: int = 100
    max_content_length: int = 10_000_000
    min_words: int = 50
    max_artifact_ratio: float = 0.05
    min_readability_score: int = 20
    min_structure_score: int = 50

@dataclass
class CleaningConfig:
    """Markdown cleaning configuration"""
    aggressive_cleaning: bool = True
    preserve_tables: bool = True
    min_line_length: int = 3
    remove_html_tags: bool = True
    normalize_whitespace: bool = True

@dataclass
class ChunkingConfig:
    """Chunking behavior configuration"""
    target_llm: LLMModel = LLMModel.CUSTOM
    chunk_size: int = 4000
    overlap: int = 200
    min_chunk_size: int = 1000
    max_chunk_size: int = 8000
    strategy: str = "semantic"  # semantic, fixed_size, sliding_window

@dataclass
class ConversionConfig:
    """Document conversion configuration"""
    pdf_converter: str = "local_marker"  # local_marker, cloud_marker
    document_converter: str = "pandoc"  # pandoc
    marker_cloud_api_key_env: str = "MARKER_API_KEY"
    marker_local_base_url: str = "http://localhost:8000"
    marker_cloud_base_url: str = "https://www.datalab.to/api/v1"
    pandoc_options: str = "--wrap=none --strip-comments --markdown-headings=atx"
    supported_pdf_formats: List[str] = None
    supported_document_formats: List[str] = None
    conversion_timeout: int = 3600
    max_retries: int = 3
    retry_delay: int = 10
    # Marker options
    use_llm: bool = False
    force_ocr: bool = False
    paginate: bool = False
    strip_existing_ocr: bool = True
    disable_image_extraction: bool = True
    # Cloud polling
    max_polls: int = 300
    poll_interval: int = 2
    
    def __post_init__(self):
        if self.supported_pdf_formats is None:
            self.supported_pdf_formats = ["pdf"]
        if self.supported_document_formats is None:
            self.supported_document_formats = ["epub", "mobi", "azw", "azw3", "html", "htm", "docx", "rtf"]

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    directories: DirectoryConfig
    llm_presets: Dict[str, LLMPreset]
    validation: ValidationThresholds
    cleaning: CleaningConfig
    chunking: ChunkingConfig
    conversion: ConversionConfig
    skip_existing: bool = True
    max_workers: int = 3
    log_level: str = "INFO"

class ConfigManager:
    """Configuration manager with file loading/saving capabilities"""
    
    DEFAULT_LLM_PRESETS = {
        "gpt-3.5-turbo": LLMPreset(3000, 150, "Cost-effective processing"),
        "gpt-4": LLMPreset(6000, 300, "Balanced performance"),
        "claude-3": LLMPreset(8000, 400, "Large context tasks"),
        "llama-2": LLMPreset(3500, 175, "Open-source deployment"),
        "gemini-pro": LLMPreset(7000, 350, "Google's model"),
        "custom": LLMPreset(4000, 200, "Custom configuration")
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("pipeline_config.json")
        self._config: Optional[PipelineConfig] = None
    
    @property
    def config(self) -> PipelineConfig:
        """Get current configuration, loading from file if needed"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self) -> PipelineConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return self._create_default_config()
    
    def save_config(self, config: Optional[PipelineConfig] = None) -> None:
        """Save configuration to file"""
        config = config or self.config
        data = self._config_to_dict(config)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to {self.config_path}")
    
    def _create_default_config(self) -> PipelineConfig:
        """Create default configuration"""
        return PipelineConfig(
            directories=DirectoryConfig(),
            llm_presets=self.DEFAULT_LLM_PRESETS,
            validation=ValidationThresholds(),
            cleaning=CleaningConfig(),
            chunking=ChunkingConfig(),
            conversion=ConversionConfig()
        )
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization"""
        return {
            "directories": asdict(config.directories),
            "llm_presets": {
                name: asdict(preset) for name, preset in config.llm_presets.items()
            },
            "validation_thresholds": asdict(config.validation),
            "cleaning_settings": asdict(config.cleaning),
            "chunking_settings": {
                **asdict(config.chunking),
                "target_llm": config.chunking.target_llm.value
            },
            "conversion_settings": asdict(config.conversion),
            "pipeline_settings": {
                "skip_existing": config.skip_existing,
                "max_workers": config.max_workers,
                "log_level": config.log_level
            }
        }
    
    def _dict_to_config(self, data: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to config object"""
        # Parse LLM presets
        llm_presets = {}
        for name, preset_data in data.get("llm_presets", {}).items():
            llm_presets[name] = LLMPreset(**preset_data)
        
        # Parse chunking config
        chunking_data = data.get("chunking_settings", {})
        if "target_llm" in chunking_data:
            chunking_data["target_llm"] = LLMModel(chunking_data["target_llm"])
        
        # Parse conversion config
        conversion_data = data.get("conversion_settings", {})
        
        # Parse pipeline settings
        pipeline_settings = data.get("pipeline_settings", {})
        
        return PipelineConfig(
            directories=DirectoryConfig(**data.get("directories", {})),
            llm_presets=llm_presets or self.DEFAULT_LLM_PRESETS,
            validation=ValidationThresholds(**data.get("validation_thresholds", {})),
            cleaning=CleaningConfig(**data.get("cleaning_settings", {})),
            chunking=ChunkingConfig(**chunking_data),
            conversion=ConversionConfig(**conversion_data),
            skip_existing=pipeline_settings.get("skip_existing", True),
            max_workers=pipeline_settings.get("max_workers", 3),
            log_level=pipeline_settings.get("log_level", "INFO")
        )
    
    def update_llm_settings(self, llm_model: str) -> None:
        """Update chunking settings based on LLM model"""
        if llm_model in self.config.llm_presets:
            preset = self.config.llm_presets[llm_model]
            self.config.chunking.chunk_size = preset.chunk_size
            self.config.chunking.overlap = preset.overlap
            self.config.chunking.target_llm = LLMModel(llm_model)
    
    def get_directory_path(self, directory_type: str) -> Path:
        """Get path for specific directory type"""
        directories = asdict(self.config.directories)
        if directory_type not in directories:
            raise ValueError(f"Unknown directory type: {directory_type}")
        return Path(directories[directory_type])
    
    def create_directories(self) -> None:
        """Create all configured directories"""
        for directory in asdict(self.config.directories).values():
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global configuration instance
_config_manager = None

def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> PipelineConfig:
    """Get current pipeline configuration"""
    return get_config_manager().config

# Environment variable overrides
def apply_env_overrides(config: PipelineConfig) -> PipelineConfig:
    """Apply environment variable overrides to configuration"""
    # Allow environment variables to override key settings
    if "CHUNK_SIZE" in os.environ:
        config.chunking.chunk_size = int(os.environ["CHUNK_SIZE"])
    
    if "OVERLAP_SIZE" in os.environ:
        config.chunking.overlap = int(os.environ["OVERLAP_SIZE"])
    
    if "TARGET_LLM" in os.environ:
        config.chunking.target_llm = LLMModel(os.environ["TARGET_LLM"])
    
    if "MAX_WORKERS" in os.environ:
        config.max_workers = int(os.environ["MAX_WORKERS"])
    
    if "LOG_LEVEL" in os.environ:
        config.log_level = os.environ["LOG_LEVEL"]
    
    if "CONVERSION_TIMEOUT" in os.environ:
        config.conversion.conversion_timeout = int(os.environ["CONVERSION_TIMEOUT"])
    # Marker toggles via env
    env_bools = {
        "MARKER_USE_LLM": ("use_llm", bool),
        "MARKER_FORCE_OCR": ("force_ocr", bool),
        "MARKER_PAGINATE": ("paginate", bool),
        "MARKER_STRIP_EXISTING_OCR": ("strip_existing_ocr", bool),
        "MARKER_DISABLE_IMAGE_EXTRACTION": ("disable_image_extraction", bool),
    }
    for env_key, (attr, _typ) in env_bools.items():
        if env_key in os.environ:
            val = os.environ[env_key].strip().lower() in ("1", "true", "yes", "on")
            setattr(config.conversion, attr, val)

    if "MAX_POLLS" in os.environ:
        config.conversion.max_polls = int(os.environ["MAX_POLLS"])
    if "POLL_INTERVAL" in os.environ:
        config.conversion.poll_interval = int(os.environ["POLL_INTERVAL"])

    return config

if __name__ == "__main__":
    # CLI for configuration management
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management")
    parser.add_argument("--create-default", action="store_true", 
                       help="Create default configuration file")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing configuration")
    parser.add_argument("--show", action="store_true",
                       help="Show current configuration")
    
    args = parser.parse_args()
    
    config_manager = get_config_manager()
    
    if args.create_default:
        config_manager.save_config()
        print("Default configuration created")
    
    if args.validate:
        try:
            config = config_manager.load_config()
            print("✓ Configuration is valid")
        except Exception as e:
            print(f"✗ Configuration error: {e}")
    
    if args.show:
        config = config_manager.config
        print(json.dumps(config_manager._config_to_dict(config), indent=2))
