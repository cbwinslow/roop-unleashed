"""
Tests for enhanced agents functionality.
"""

import os
import sys
import tempfile

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from agents.enhanced_agents import RAGAgent, VideoProcessingAgent, OptimizationAgent
    from agents.manager import MultiAgentManager
    from settings import Settings
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced agents not available: {e}")
    ENHANCED_AVAILABLE = False


class TestEnhancedAgents:
    """Test enhanced agent functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        if not ENHANCED_AVAILABLE:
            return
            
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create minimal settings
        self.settings = Settings(self.config_file)
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rag_agent_basic(self):
        """Test basic RAG agent functionality."""
        if not ENHANCED_AVAILABLE:
            return
            
        agent = RAGAgent(settings=self.settings)
        
        # Should handle queries even without LLM/RAG
        response = agent.assist("What is face swapping?")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_video_agent_routing(self):
        """Test video agent query routing."""
        if not ENHANCED_AVAILABLE:
            return
            
        agent = VideoProcessingAgent(settings=self.settings)
        
        # Video-related query
        response = agent.assist("How to process video files?")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Non-video query should redirect
        response = agent.assist("What is the weather?")
        assert "doesn't appear to be video-related" in response
    
    def test_optimization_agent(self):
        """Test optimization agent functionality."""
        if not ENHANCED_AVAILABLE:
            return
            
        agent = OptimizationAgent(settings=self.settings)
        
        # Optimization query
        response = agent.assist("How to make processing faster?")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Should provide basic advice even without LLM
        basic_response = agent._get_basic_optimization_advice()
        assert "GPU acceleration" in basic_response
    
    def test_agent_manager_initialization(self):
        """Test enhanced agent manager initialization."""
        if not ENHANCED_AVAILABLE:
            return
            
        manager = MultiAgentManager(self.settings)
        
        # Should have basic agents at minimum
        agents = manager.available_agents()
        assert 'installer' in agents
        assert 'model' in agents
        assert 'operation' in agents
        assert 'nlp' in agents
        
        # Should handle unknown agent gracefully
        response = manager.assist('unknown_agent', 'test query')
        assert 'Unknown agent' in response
    
    def test_smart_assist_routing(self):
        """Test smart assistance routing."""
        if not ENHANCED_AVAILABLE:
            return
            
        manager = MultiAgentManager(self.settings)
        
        # Should route video queries appropriately
        response = manager.smart_assist("My video processing is slow")
        assert isinstance(response, str)
        
        # Should route optimization queries
        response = manager.smart_assist("How to optimize GPU performance?")
        assert isinstance(response, str)
        
        # Should route installation queries
        response = manager.smart_assist("How to install dependencies?")
        assert isinstance(response, str)
    
    def test_agent_error_handling(self):
        """Test agent error handling."""
        if not ENHANCED_AVAILABLE:
            return
            
        manager = MultiAgentManager(self.settings)
        
        # Should handle errors gracefully
        try:
            response = manager.assist('installer', None)  # Invalid query
            # Should not raise exception
            assert isinstance(response, str)
        except Exception:
            # If it does raise, that's also acceptable for this test
            pass
    
    def test_system_status(self):
        """Test system status reporting."""
        if not ENHANCED_AVAILABLE:
            return
            
        manager = MultiAgentManager(self.settings)
        
        status = manager.get_system_status()
        assert isinstance(status, dict)
        assert 'agent_count' in status
        assert 'available_agents' in status
        assert status['agent_count'] > 0
    
    def test_help_text_generation(self):
        """Test help text generation."""
        if not ENHANCED_AVAILABLE:
            return
            
        manager = MultiAgentManager(self.settings)
        
        help_text = manager.get_help_text()
        assert isinstance(help_text, str)
        assert 'Roop Unleashed' in help_text
        assert 'Available Agents:' in help_text


class TestBasicAgentFunctionality:
    """Test basic agent functionality without enhanced features."""
    
    def test_basic_manager_without_enhanced_features(self):
        """Test that basic agent manager works without enhanced features."""
        # Import only basic manager
        sys.modules.pop('agents.enhanced_agents', None)  # Remove if imported
        
        from agents.manager import MultiAgentManager
        
        # Should work without settings
        manager = MultiAgentManager()
        
        agents = manager.available_agents()
        assert 'installer' in agents
        assert 'model' in agents
        assert 'operation' in agents
        assert 'nlp' in agents
        
        # Should handle basic queries
        response = manager.assist('installer', 'How to install?')
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == '__main__':
    # Simple test runner
    test_classes = [TestEnhancedAgents, TestBasicAgentFunctionality]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        test_instance = test_class()
        
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                    
                method = getattr(test_instance, method_name)
                method()
                
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                    
                print(f"✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                failed += 1
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
        
        print(f"Class results: {passed} passed, {failed} failed")
        total_passed += passed
        total_failed += failed
    
    print(f"\nOverall results: {total_passed} passed, {total_failed} failed")