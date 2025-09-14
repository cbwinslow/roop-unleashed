import yaml
import os
import logging
import shutil
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self, config_file: str, defaults_file: str = "config_defaults.yaml"):
        self.config_file = config_file
        self.defaults_file = defaults_file
        self._ensure_config_exists()
        self.load()

    def _ensure_config_exists(self) -> None:
        """Ensure configuration file exists, create from defaults if not."""
        if not os.path.exists(self.config_file):
            logger.info(f"Configuration file {self.config_file} not found, creating from defaults...")
            try:
                if os.path.exists(self.defaults_file):
                    shutil.copy2(self.defaults_file, self.config_file)
                    logger.info(f"Created configuration file from {self.defaults_file}")
                else:
                    # Create minimal config if defaults don't exist
                    self._create_minimal_config()
            except Exception as e:
                logger.error(f"Failed to create configuration file: {e}")
                self._create_minimal_config()

    def _create_minimal_config(self) -> None:
        """Create a minimal configuration file."""
        minimal_config = {
            'selected_theme': "Default",
            'server_name': "",
            'server_port': 0,
            'server_share': False,
            'output_image_format': 'png',
            'output_video_format': 'mp4',
            'output_video_codec': 'libx264',
            'video_quality': 14,
            'clear_output': True,
            'live_cam_start_active': False,
            'max_threads': 4,
            'memory_limit': 0,
            'frame_buffer_size': 4,
            'provider': 'cuda',
            'force_cpu': False
        }
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(minimal_config, f, default_flow_style=False)
            logger.info("Created minimal configuration file")
        except Exception as e:
            logger.error(f"Failed to create minimal configuration: {e}")

    def default_get(self, data: Optional[Dict], name: str, default: Any) -> Any:
        """Safely get a value from config data with default fallback."""
        if data is None:
            return default
        
        try:
            # Support nested keys like 'ai_config.local_llm.enabled'
            if '.' in name:
                value = data
                for key in name.split('.'):
                    value = value.get(key, {})
                return value if value != {} else default
            else:
                return data.get(name, default)
        except Exception as e:
            logger.warning(f"Error getting config value '{name}': {e}")
            return default

    def load(self) -> None:
        """Load configuration from file with comprehensive error handling."""
        data = None
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {self.config_file}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_file} not found")
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")

        # Load basic settings with fallbacks
        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "")
        self.server_port = self.default_get(data, 'server_port', 0)
        self.server_share = self.default_get(data, 'server_share', False)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.video_quality = self.default_get(data, 'video_quality', 14)
        self.clear_output = self.default_get(data, 'clear_output', True)
        self.live_cam_start_active = self.default_get(data, 'live_cam_start_active', False)
        self.max_threads = self.default_get(data, 'max_threads', 4)
        self.memory_limit = self.default_get(data, 'memory_limit', 0)
        self.frame_buffer_size = self.default_get(data, 'frame_buffer_size', 4)
        self.provider = self.default_get(data, 'provider', 'cuda')
        self.force_cpu = self.default_get(data, 'force_cpu', False)
        
        # Load enhanced AI configuration
        self.ai_config = self.default_get(data, 'ai_config', {})
        self.agents = self.default_get(data, 'agents', {})
        self.nvidia_config = self.default_get(data, 'nvidia_config', {})
        self.error_handling = self.default_get(data, 'error_handling', {})
        self.logging_config = self.default_get(data, 'logging', {})
        self.security = self.default_get(data, 'security', {})


    def save(self) -> None:
        """Save current configuration to file with error handling."""
        try:
            data = {
                'selected_theme': self.selected_theme,
                'server_name': self.server_name,
                'server_port': self.server_port,
                'server_share': self.server_share,
                'output_image_format': self.output_image_format,
                'output_video_format': self.output_video_format,
                'output_video_codec': self.output_video_codec,
                'video_quality': self.video_quality,
                'clear_output': self.clear_output,
                'live_cam_start_active': self.live_cam_start_active,
                'max_threads': self.max_threads,
                'memory_limit': self.memory_limit,
                'frame_buffer_size': self.frame_buffer_size,
                'provider': self.provider,
                'force_cpu': self.force_cpu,
                'ai_config': self.ai_config,
                'agents': self.agents,
                'nvidia_config': self.nvidia_config,
                'error_handling': self.error_handling,
                'logging': self.logging_config,
                'security': self.security
            }
            
            # Create backup of existing config
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup"
                shutil.copy2(self.config_file, backup_file)
                logger.debug(f"Created configuration backup: {backup_file}")
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate basic types and ranges
            if not isinstance(self.max_threads, int) or self.max_threads < 1:
                logger.warning("Invalid max_threads, setting to 4")
                self.max_threads = 4
            
            if not isinstance(self.video_quality, int) or not (0 <= self.video_quality <= 51):
                logger.warning("Invalid video_quality, setting to 14")
                self.video_quality = 14
            
            if self.provider not in ['cuda', 'rocm', 'directml', 'cpu']:
                logger.warning(f"Invalid provider '{self.provider}', setting to 'cuda'")
                self.provider = 'cuda'
                
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_ai_setting(self, key: str, default: Any = None) -> Any:
        """Get AI-specific configuration setting."""
        return self.default_get(self.ai_config, key, default)

    def get_nvidia_setting(self, key: str, default: Any = None) -> Any:
        """Get NVIDIA-specific configuration setting."""
        return self.default_get(self.nvidia_config, key, default)

    def get_agent_setting(self, agent_name: str, key: str, default: Any = None) -> Any:
        """Get agent-specific configuration setting."""
        agent_config = self.agents.get(agent_name, {})
        return agent_config.get(key, default)
