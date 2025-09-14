"""
Configuration migration utility for roop-unleashed.
Helps migrate from old config format to new enhanced format.
"""

import os
import yaml
import logging
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Migrates configuration from old format to new enhanced format."""
    
    def __init__(self, old_config_path: str, new_config_path: str, defaults_path: str):
        self.old_config_path = Path(old_config_path)
        self.new_config_path = Path(new_config_path)
        self.defaults_path = Path(defaults_path)
    
    def needs_migration(self) -> bool:
        """Check if migration is needed."""
        if not self.old_config_path.exists():
            return False
        
        if not self.new_config_path.exists():
            return True
        
        # Check if new config has enhanced structure
        try:
            with open(self.new_config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # If it has ai_config, it's already migrated
            return 'ai_config' not in new_config
        except Exception:
            return True
    
    def migrate(self) -> bool:
        """Perform the migration."""
        try:
            logger.info(f"Migrating configuration from {self.old_config_path} to {self.new_config_path}")
            
            # Load old configuration
            old_config = self._load_old_config()
            if old_config is None:
                return False
            
            # Load defaults for new structure
            defaults = self._load_defaults()
            if defaults is None:
                return False
            
            # Create backup of old config
            backup_path = f"{self.old_config_path}.backup"
            shutil.copy2(self.old_config_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Merge old config with new defaults
            new_config = self._merge_configs(old_config, defaults)
            
            # Save new configuration
            with open(self.new_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _load_old_config(self) -> Optional[Dict[str, Any]]:
        """Load old configuration file."""
        try:
            with open(self.old_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load old config: {e}")
            return None
    
    def _load_defaults(self) -> Optional[Dict[str, Any]]:
        """Load default configuration structure."""
        try:
            with open(self.defaults_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load defaults: {e}")
            return None
    
    def _merge_configs(self, old_config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Merge old configuration with new defaults structure."""
        new_config = defaults.copy()
        
        # Map old settings to new structure
        setting_mappings = {
            'selected_theme': 'selected_theme',
            'server_name': 'server_name',
            'server_port': 'server_port',
            'server_share': 'server_share',
            'output_image_format': 'output_image_format',
            'output_video_format': 'output_video_format',
            'output_video_codec': 'output_video_codec',
            'video_quality': 'video_quality',
            'clear_output': 'clear_output',
            'live_cam_start_active': 'live_cam_start_active',
            'max_threads': 'max_threads',
            'memory_limit': 'memory_limit',
            'frame_buffer_size': 'frame_buffer_size',
            'provider': 'provider',
            'force_cpu': 'force_cpu'
        }
        
        # Apply mappings
        for old_key, new_key in setting_mappings.items():
            if old_key in old_config:
                new_config[new_key] = old_config[old_key]
        
        # Set some defaults based on old config
        if old_config.get('provider') == 'cuda':
            new_config['nvidia_config']['tensorrt']['enabled'] = True
        
        # Enable basic AI features if GPU is available
        if old_config.get('provider') in ['cuda', 'rocm']:
            new_config['ai_config']['local_llm']['enabled'] = False  # User can enable manually
            new_config['agents']['optimization_agent']['enabled'] = True
        
        return new_config
    
    def get_migration_report(self) -> str:
        """Generate a migration report."""
        if not self.needs_migration():
            return "No migration needed - configuration is already up to date."
        
        if not self.old_config_path.exists():
            return "No old configuration file found."
        
        try:
            old_config = self._load_old_config()
            if not old_config:
                return "Could not read old configuration file."
            
            report_lines = [
                "Configuration Migration Report",
                "=" * 35,
                "",
                f"Old config: {self.old_config_path}",
                f"New config: {self.new_config_path}",
                "",
                "Settings to migrate:",
            ]
            
            setting_mappings = {
                'selected_theme': 'UI Theme',
                'provider': 'GPU Provider',
                'max_threads': 'Max Threads',
                'video_quality': 'Video Quality',
                'server_port': 'Server Port'
            }
            
            for key, description in setting_mappings.items():
                if key in old_config:
                    value = old_config[key]
                    report_lines.append(f"  {description}: {value}")
            
            report_lines.extend([
                "",
                "New features to be added:",
                "  - Enhanced AI agent system",
                "  - Local LLM integration support",
                "  - RAG knowledge base",
                "  - NVIDIA optimization settings",
                "  - Comprehensive error handling",
                "",
                "To migrate, run: python -c \"from roop.config_migrator import migrate_config; migrate_config()\""
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Error generating migration report: {e}"


def migrate_config(old_path: str = "config.yaml", 
                  new_path: str = "config.yaml", 
                  defaults_path: str = "config_defaults.yaml") -> bool:
    """Convenience function to migrate configuration."""
    migrator = ConfigMigrator(old_path, new_path, defaults_path)
    
    if not migrator.needs_migration():
        print("Configuration is already up to date.")
        return True
    
    print("Starting configuration migration...")
    print(migrator.get_migration_report())
    print()
    
    success = migrator.migrate()
    if success:
        print("Migration completed successfully!")
        print("Your old configuration has been backed up.")
        print("You can now take advantage of the new AI features.")
    else:
        print("Migration failed. Please check the logs for details.")
    
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        migrator = ConfigMigrator("config.yaml", "config.yaml", "config_defaults.yaml")
        print(migrator.get_migration_report())
    else:
        migrate_config()