"""
Tests for enhanced configuration system.
"""

import os
import tempfile
import yaml
from pathlib import Path

# Add the project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from settings import Settings


class TestEnhancedSettings:
    """Test the enhanced settings system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        self.defaults_file = os.path.join(self.temp_dir, 'test_defaults.yaml')
        
        # Create a test defaults file
        defaults = {
            'selected_theme': 'Default',
            'max_threads': 4,
            'provider': 'cuda',
            'ai_config': {
                'local_llm': {
                    'enabled': False,
                    'model': 'test_model'
                }
            }
        }
        
        with open(self.defaults_file, 'w') as f:
            yaml.dump(defaults, f)
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation_from_defaults(self):
        """Test that config is created from defaults when missing."""
        settings = Settings(self.config_file, self.defaults_file)
        
        # Config file should be created
        assert os.path.exists(self.config_file)
        
        # Should have default values
        assert settings.selected_theme == 'Default'
        assert settings.max_threads == 4
        assert settings.provider == 'cuda'
    
    def test_config_loading_existing(self):
        """Test loading existing configuration."""
        # Create existing config
        existing_config = {
            'selected_theme': 'Dark',
            'max_threads': 8,
            'provider': 'cpu'
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(existing_config, f)
        
        settings = Settings(self.config_file, self.defaults_file)
        
        assert settings.selected_theme == 'Dark'
        assert settings.max_threads == 8
        assert settings.provider == 'cpu'
    
    def test_ai_setting_access(self):
        """Test accessing AI-specific settings."""
        settings = Settings(self.config_file, self.defaults_file)
        
        # Should access nested AI settings
        assert settings.get_ai_setting('local_llm.enabled', True) == False
        assert settings.get_ai_setting('local_llm.model', 'default') == 'test_model'
        assert settings.get_ai_setting('nonexistent.key', 'fallback') == 'fallback'
    
    def test_config_validation(self):
        """Test configuration validation."""
        settings = Settings(self.config_file, self.defaults_file)
        
        # Set invalid values
        settings.max_threads = -1
        settings.video_quality = 100
        settings.provider = 'invalid_provider'
        
        # Validation should fix these
        assert settings.validate() == True
        assert settings.max_threads == 4  # Fixed to default
        assert settings.video_quality == 14  # Fixed to default
        assert settings.provider == 'cuda'  # Fixed to default
    
    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        settings = Settings(self.config_file, self.defaults_file)
        
        # Modify settings
        settings.selected_theme = 'Custom'
        settings.max_threads = 6
        settings.ai_config = {'test': 'value'}
        
        # Save
        settings.save()
        
        # Load new instance
        new_settings = Settings(self.config_file, self.defaults_file)
        
        assert new_settings.selected_theme == 'Custom'
        assert new_settings.max_threads == 6
        assert new_settings.ai_config == {'test': 'value'}
    
    def test_backup_creation(self):
        """Test that backup is created when saving."""
        # Create initial config
        settings = Settings(self.config_file, self.defaults_file)
        settings.save()
        
        # Modify and save again
        settings.selected_theme = 'Modified'
        settings.save()
        
        # Backup should exist
        backup_file = f"{self.config_file}.backup"
        assert os.path.exists(backup_file)
    
    def test_minimal_config_fallback(self):
        """Test fallback to minimal config when defaults are missing."""
        # Remove defaults file
        os.remove(self.defaults_file)
        
        settings = Settings(self.config_file, self.defaults_file)
        
        # Should still work with minimal config
        assert settings.selected_theme == 'Default'
        assert settings.provider == 'cuda'
        assert os.path.exists(self.config_file)


if __name__ == '__main__':
    # Simple test runner
    test_instance = TestEnhancedSettings()
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            test_instance.teardown_method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
            test_instance.teardown_method()
    
    print(f"\nResults: {passed} passed, {failed} failed")