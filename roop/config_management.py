#!/usr/bin/env python3
"""
Configuration management system for optimization parameters and AI agent settings.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for face processing parameters."""
    # Face detection settings
    face_detection_threshold: float = 0.7
    face_detection_scale_factor: float = 1.1
    face_detection_min_neighbors: int = 5
    multi_scale_detection: bool = True
    adaptive_threshold: bool = True
    
    # Face swapping settings
    swap_quality_threshold: float = 0.8
    blend_method: str = "seamless"  # poisson, multi_band, gradient, seamless
    face_enhancement: bool = True
    preserve_facial_features: bool = True
    include_hair_region: bool = True
    
    # Performance settings
    batch_size: int = 4
    max_resolution: int = 1920
    gpu_memory_limit_mb: int = 6144
    cpu_thread_count: int = 0  # 0 = auto-detect
    processing_timeout_seconds: int = 120
    
    # Quality settings
    output_quality: float = 0.9
    ssim_threshold: float = 0.75
    psnr_threshold: float = 25.0
    temporal_consistency_weight: float = 0.3


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and telemetry."""
    # Metrics collection
    metrics_collection_interval: float = 1.0
    metrics_buffer_size: int = 10000
    enable_system_metrics: bool = True
    enable_gpu_metrics: bool = True
    enable_application_metrics: bool = True
    
    # Alert settings
    enable_alerting: bool = True
    alert_cooldown_seconds: int = 300
    alert_email_notifications: bool = False
    alert_webhook_url: Optional[str] = None
    
    # Performance thresholds
    cpu_usage_alert_threshold: float = 85.0
    memory_usage_alert_threshold: float = 90.0
    gpu_usage_alert_threshold: float = 95.0
    processing_time_alert_threshold: float = 10.0  # seconds
    error_rate_alert_threshold: float = 0.1  # 10%
    
    # Quality thresholds
    quality_alert_threshold: float = 0.7
    ssim_alert_threshold: float = 0.6
    psnr_alert_threshold: float = 20.0
    
    # Telemetry export
    enable_telemetry_export: bool = False
    telemetry_export_interval: int = 300  # seconds
    telemetry_export_format: str = "json"  # json, prometheus, influxdb
    telemetry_export_path: str = "/tmp/roop_telemetry"


@dataclass
class OptimizationConfig:
    """Configuration for AI-driven optimization."""
    # Optimization engine settings
    enable_optimization: bool = True
    optimization_interval: float = 30.0  # seconds
    optimization_aggressiveness: str = "balanced"  # conservative, balanced, aggressive
    
    # Auto-tuning settings
    enable_auto_batch_sizing: bool = True
    enable_auto_quality_adjustment: bool = True
    enable_auto_resource_allocation: bool = True
    enable_auto_error_recovery: bool = True
    
    # Learning parameters
    learning_rate: float = 0.1
    adaptation_window_size: int = 100  # Number of samples for learning
    performance_target_fps: float = 5.0
    quality_target_ssim: float = 0.85
    
    # Resource limits
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 85.0
    max_gpu_memory_usage_percent: float = 90.0
    
    # Optimization rules
    batch_size_min: int = 1
    batch_size_max: int = 16
    quality_threshold_min: float = 0.6
    quality_threshold_max: float = 0.95
    
    # Recovery settings
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: int = 60
    fallback_to_cpu: bool = True


@dataclass
class AgentConfig:
    """Configuration for AI agent system."""
    # Agent system settings
    enable_agent_system: bool = True
    max_concurrent_agents: int = 5
    agent_communication_timeout: float = 30.0
    agent_heartbeat_interval: float = 10.0
    
    # Individual agent settings
    enable_monitoring_agent: bool = True
    enable_optimization_agent: bool = True
    enable_quality_agent: bool = True
    enable_error_recovery_agent: bool = True
    enable_learning_agent: bool = True
    
    # NLP agent settings
    enable_nlp_agent: bool = False
    nlp_model_name: str = "gpt-3.5-turbo"
    nlp_response_timeout: float = 10.0
    
    # MCP server settings
    enable_mcp_server: bool = False
    mcp_server_port: int = 8080
    mcp_server_host: str = "localhost"
    
    # Agent collaboration
    enable_agent_coordination: bool = True
    coordination_strategy: str = "hierarchical"  # flat, hierarchical, mesh
    priority_levels: List[str] = field(default_factory=lambda: ["critical", "high", "medium", "low"])


@dataclass
class UIConfig:
    """Configuration for user interface."""
    # Interface settings
    theme: str = "dark"  # dark, light, auto
    default_language: str = "en"
    enable_advanced_controls: bool = False
    show_performance_metrics: bool = True
    
    # Processing display
    show_real_time_preview: bool = True
    preview_update_interval: float = 0.5
    show_quality_metrics: bool = True
    show_processing_logs: bool = False
    
    # Input/output settings
    default_input_format: str = "auto"
    default_output_format: str = "mp4"
    default_output_quality: str = "high"
    preserve_metadata: bool = True
    
    # Performance display
    show_fps_counter: bool = True
    show_memory_usage: bool = True
    show_gpu_info: bool = True
    enable_diagnostics_panel: bool = False


class ConfigurationManager:
    """Manages all configuration settings with validation and persistence."""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = Path.home() / ".roop-unleashed"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration files
        self.processing_config_file = self.config_dir / "processing.yaml"
        self.monitoring_config_file = self.config_dir / "monitoring.yaml"
        self.optimization_config_file = self.config_dir / "optimization.yaml"
        self.agent_config_file = self.config_dir / "agents.yaml"
        self.ui_config_file = self.config_dir / "ui.yaml"
        
        # Configuration instances
        self.processing = ProcessingConfig()
        self.monitoring = MonitoringConfig()
        self.optimization = OptimizationConfig()
        self.agents = AgentConfig()
        self.ui = UIConfig()
        
        # Load existing configurations
        self.load_all_configs()
        
        # Validation rules
        self.validation_rules = self._setup_validation_rules()
        
    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Setup validation rules for configuration parameters."""
        return {
            "processing": {
                "face_detection_threshold": {"min": 0.1, "max": 1.0},
                "swap_quality_threshold": {"min": 0.1, "max": 1.0},
                "batch_size": {"min": 1, "max": 32},
                "max_resolution": {"min": 256, "max": 4096},
                "gpu_memory_limit_mb": {"min": 512, "max": 24576},
                "output_quality": {"min": 0.1, "max": 1.0},
                "ssim_threshold": {"min": 0.0, "max": 1.0},
                "psnr_threshold": {"min": 10.0, "max": 50.0}
            },
            "monitoring": {
                "metrics_collection_interval": {"min": 0.1, "max": 60.0},
                "metrics_buffer_size": {"min": 100, "max": 100000},
                "cpu_usage_alert_threshold": {"min": 50.0, "max": 100.0},
                "memory_usage_alert_threshold": {"min": 50.0, "max": 100.0},
                "processing_time_alert_threshold": {"min": 1.0, "max": 300.0},
                "error_rate_alert_threshold": {"min": 0.01, "max": 1.0}
            },
            "optimization": {
                "optimization_interval": {"min": 1.0, "max": 300.0},
                "learning_rate": {"min": 0.001, "max": 1.0},
                "performance_target_fps": {"min": 0.1, "max": 60.0},
                "quality_target_ssim": {"min": 0.5, "max": 1.0},
                "max_cpu_usage_percent": {"min": 50.0, "max": 100.0},
                "max_memory_usage_percent": {"min": 50.0, "max": 100.0},
                "batch_size_min": {"min": 1, "max": 16},
                "batch_size_max": {"min": 1, "max": 32}
            }
        }
        
    def load_all_configs(self):
        """Load all configuration files."""
        try:
            self.processing = self._load_config(ProcessingConfig, self.processing_config_file)
            self.monitoring = self._load_config(MonitoringConfig, self.monitoring_config_file)
            self.optimization = self._load_config(OptimizationConfig, self.optimization_config_file)
            self.agents = self._load_config(AgentConfig, self.agent_config_file)
            self.ui = self._load_config(UIConfig, self.ui_config_file)
            
            logger.info("All configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            
    def save_all_configs(self):
        """Save all configurations to files."""
        try:
            self._save_config(self.processing, self.processing_config_file)
            self._save_config(self.monitoring, self.monitoring_config_file)
            self._save_config(self.optimization, self.optimization_config_file)
            self._save_config(self.agents, self.agent_config_file)
            self._save_config(self.ui, self.ui_config_file)
            
            logger.info("All configurations saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configurations: {e}")
            
    def _load_config(self, config_class, config_file: Path):
        """Load a specific configuration from file."""
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                if config_data:
                    # Create instance with loaded data
                    return config_class(**config_data)
                    
            except Exception as e:
                logger.warning(f"Error loading {config_file}: {e}")
                
        # Return default instance if loading fails
        return config_class()
        
    def _save_config(self, config_instance, config_file: Path):
        """Save a configuration instance to file."""
        config_data = asdict(config_instance)
        
        # Add metadata
        config_data["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
    def validate_config(self, config_name: str, config_instance) -> List[str]:
        """
        Validate a configuration instance.
        
        Args:
            config_name: Name of the configuration (e.g., 'processing')
            config_instance: Configuration instance to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if config_name not in self.validation_rules:
            return errors
            
        config_dict = asdict(config_instance)
        rules = self.validation_rules[config_name]
        
        for field_name, field_value in config_dict.items():
            if field_name in rules:
                rule = rules[field_name]
                
                # Check numeric ranges
                if isinstance(field_value, (int, float)):
                    if "min" in rule and field_value < rule["min"]:
                        errors.append(f"{field_name}: {field_value} is below minimum {rule['min']}")
                    if "max" in rule and field_value > rule["max"]:
                        errors.append(f"{field_name}: {field_value} is above maximum {rule['max']}")
                        
                # Check string values
                if isinstance(field_value, str) and "values" in rule:
                    if field_value not in rule["values"]:
                        errors.append(f"{field_name}: '{field_value}' not in allowed values {rule['values']}")
                        
        return errors
        
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all configurations."""
        validation_results = {}
        
        validation_results["processing"] = self.validate_config("processing", self.processing)
        validation_results["monitoring"] = self.validate_config("monitoring", self.monitoring)
        validation_results["optimization"] = self.validate_config("optimization", self.optimization)
        
        return validation_results
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all current configurations."""
        return {
            "processing": {
                "batch_size": self.processing.batch_size,
                "quality_threshold": self.processing.swap_quality_threshold,
                "blend_method": self.processing.blend_method,
                "include_hair": self.processing.include_hair_region
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.enable_system_metrics,
                "alerting_enabled": self.monitoring.enable_alerting,
                "cpu_threshold": self.monitoring.cpu_usage_alert_threshold,
                "memory_threshold": self.monitoring.memory_usage_alert_threshold
            },
            "optimization": {
                "auto_optimization": self.optimization.enable_optimization,
                "aggressiveness": self.optimization.optimization_aggressiveness,
                "target_fps": self.optimization.performance_target_fps,
                "target_quality": self.optimization.quality_target_ssim
            },
            "agents": {
                "agent_system_enabled": self.agents.enable_agent_system,
                "active_agents": sum([
                    self.agents.enable_monitoring_agent,
                    self.agents.enable_optimization_agent,
                    self.agents.enable_quality_agent,
                    self.agents.enable_error_recovery_agent,
                    self.agents.enable_learning_agent
                ])
            }
        }
        
    def apply_optimization_profile(self, profile: str):
        """
        Apply predefined optimization profiles.
        
        Args:
            profile: Profile name ('performance', 'quality', 'balanced', 'power_saving')
        """
        if profile == "performance":
            # Optimize for speed
            self.processing.batch_size = 8
            self.processing.swap_quality_threshold = 0.75
            self.processing.blend_method = "gradient"
            self.processing.face_enhancement = False
            self.optimization.optimization_aggressiveness = "aggressive"
            self.optimization.performance_target_fps = 8.0
            
        elif profile == "quality":
            # Optimize for quality
            self.processing.batch_size = 2
            self.processing.swap_quality_threshold = 0.9
            self.processing.blend_method = "seamless"
            self.processing.face_enhancement = True
            self.processing.include_hair_region = True
            self.optimization.optimization_aggressiveness = "conservative"
            self.optimization.quality_target_ssim = 0.9
            
        elif profile == "balanced":
            # Balanced performance and quality
            self.processing.batch_size = 4
            self.processing.swap_quality_threshold = 0.8
            self.processing.blend_method = "multi_band"
            self.processing.face_enhancement = True
            self.optimization.optimization_aggressiveness = "balanced"
            
        elif profile == "power_saving":
            # Minimize resource usage
            self.processing.batch_size = 1
            self.processing.max_resolution = 1280
            self.processing.face_enhancement = False
            self.optimization.max_cpu_usage_percent = 60.0
            self.optimization.max_memory_usage_percent = 70.0
            
        else:
            raise ValueError(f"Unknown profile: {profile}")
            
        logger.info(f"Applied optimization profile: {profile}")
        
    def export_config(self, export_path: str, format: str = "yaml"):
        """
        Export all configurations to a single file.
        
        Args:
            export_path: Path to export file
            format: Export format ('yaml' or 'json')
        """
        export_data = {
            "processing": asdict(self.processing),
            "monitoring": asdict(self.monitoring),
            "optimization": asdict(self.optimization),
            "agents": asdict(self.agents),
            "ui": asdict(self.ui),
            "export_metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                "format": format
            }
        }
        
        export_path = Path(export_path)
        
        if format == "yaml":
            with open(export_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)
        elif format == "json":
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Configuration exported to {export_path}")
        
    def import_config(self, import_path: str):
        """
        Import configurations from a file.
        
        Args:
            import_path: Path to import file
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
            
        try:
            if import_path.suffix.lower() in ['.yaml', '.yml']:
                with open(import_path, 'r') as f:
                    import_data = yaml.safe_load(f)
            elif import_path.suffix.lower() == '.json':
                with open(import_path, 'r') as f:
                    import_data = json.load(f)
            else:
                raise ValueError(f"Unsupported import format: {import_path.suffix}")
                
            # Apply imported configurations
            if "processing" in import_data:
                self.processing = ProcessingConfig(**import_data["processing"])
            if "monitoring" in import_data:
                self.monitoring = MonitoringConfig(**import_data["monitoring"])
            if "optimization" in import_data:
                self.optimization = OptimizationConfig(**import_data["optimization"])
            if "agents" in import_data:
                self.agents = AgentConfig(**import_data["agents"])
            if "ui" in import_data:
                self.ui = UIConfig(**import_data["ui"])
                
            # Validate imported configurations
            validation_results = self.validate_all_configs()
            
            # Report validation issues
            for config_name, errors in validation_results.items():
                if errors:
                    logger.warning(f"Validation errors in {config_name}: {errors}")
                    
            logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            raise
            
    def reset_to_defaults(self, config_type: Optional[str] = None):
        """
        Reset configurations to default values.
        
        Args:
            config_type: Specific config to reset, or None for all configs
        """
        if config_type is None or config_type == "processing":
            self.processing = ProcessingConfig()
        if config_type is None or config_type == "monitoring":
            self.monitoring = MonitoringConfig()
        if config_type is None or config_type == "optimization":
            self.optimization = OptimizationConfig()
        if config_type is None or config_type == "agents":
            self.agents = AgentConfig()
        if config_type is None or config_type == "ui":
            self.ui = UIConfig()
            
        logger.info(f"Reset {'all configurations' if config_type is None else config_type} to defaults")
        
    def create_backup(self, backup_name: Optional[str] = None):
        """Create a backup of current configurations."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_dir = self.config_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"{backup_name}.yaml"
        self.export_config(str(backup_file), "yaml")
        
        logger.info(f"Configuration backup created: {backup_file}")
        
        return backup_file
        
    def restore_backup(self, backup_name: str):
        """Restore configurations from a backup."""
        backup_dir = self.config_dir / "backups"
        backup_file = backup_dir / f"{backup_name}.yaml"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup not found: {backup_file}")
            
        self.import_config(str(backup_file))
        logger.info(f"Configuration restored from backup: {backup_name}")
        
    def list_backups(self) -> List[str]:
        """List available configuration backups."""
        backup_dir = self.config_dir / "backups"
        
        if not backup_dir.exists():
            return []
            
        backups = []
        for backup_file in backup_dir.glob("*.yaml"):
            backups.append(backup_file.stem)
            
        return sorted(backups, reverse=True)


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return config_manager


if __name__ == "__main__":
    # Example usage
    config = ConfigurationManager()
    
    # Apply a performance profile
    config.apply_optimization_profile("balanced")
    
    # Validate configurations
    validation_results = config.validate_all_configs()
    print("Validation results:", validation_results)
    
    # Get configuration summary
    summary = config.get_config_summary()
    print("Configuration summary:", json.dumps(summary, indent=2))
    
    # Create backup
    backup_file = config.create_backup("test_backup")
    print(f"Backup created: {backup_file}")
    
    # Export configuration
    config.export_config("/tmp/roop_config_export.yaml")
    print("Configuration exported")
    
    # Save all configurations
    config.save_all_configs()
    print("All configurations saved")