"""
AI Chat Interface for Roop Unleashed
Provides a chat interface to interact with AI agents for face swapping assistance.
"""

import gradio as gr
import logging
from typing import List, Tuple, Optional, Dict, Union
import time

# Import agent manager
try:
    from agents.manager import MultiAgentManager
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    logging.warning("Agent manager not available")

# Import face swapping knowledge base
try:
    from roop.face_swap_knowledge import get_knowledge_for_query, get_random_tip, HELP_CATEGORIES
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    logging.warning("Face swap knowledge base not available")

logger = logging.getLogger(__name__)


class AIChatInterface:
    """AI Chat Interface for interacting with specialized agents."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.agent_manager = None
        self.conversation_history = []
        
        # Initialize agent manager if available
        if AGENTS_AVAILABLE and settings:
            try:
                self.agent_manager = MultiAgentManager(settings)
                logger.info("AI Chat Interface initialized with agent manager")
            except Exception as e:
                logger.error(f"Failed to initialize agent manager: {e}")
        
        # Face swapping specific knowledge base
        self.face_swap_knowledge = {
            "face detection": "Roop uses InsightFace models for face detection. You can adjust detection sensitivity in Advanced settings.",
            "gpu acceleration": "Enable CUDA, ROCm, or DirectML in Settings > GPU Provider for faster processing.",
            "video quality": "Use CRF values 14-20 for good quality. Lower values = higher quality but larger files.",
            "face enhancement": "Use GFPGAN or CodeFormer for improving face quality after swapping.",
            "multiple faces": "You can swap multiple faces by selecting different target faces in the gallery.",
            "troubleshooting": "Check GPU memory usage, reduce resolution, or switch to CPU mode if encountering errors."
        }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available AI agents."""
        if self.agent_manager:
            return self.agent_manager.available_agents()
        return ["general"]
    
    def chat_response(self, message: str, agent: str, history: List[dict]) -> Tuple[List[dict], str]:
        """Generate chat response using selected agent."""
        if not message.strip():
            return history, ""
        
        # Check for face swapping specific queries first
        response = self._check_face_swap_knowledge(message)
        
        if not response and self.agent_manager and agent != "general":
            try:
                if agent == "auto":
                    response = self.agent_manager.smart_assist(message)
                else:
                    response = self.agent_manager.assist(agent, message)
            except Exception as e:
                response = f"Error from {agent} agent: {str(e)}"
        
        # Fallback to general knowledge
        if not response:
            response = self._general_response(message)
        
        # Add to history in messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        self.conversation_history = history
        
        return history, ""
    
    def _check_face_swap_knowledge(self, message: str) -> Optional[str]:
        """Check if message matches face swapping knowledge base."""
        # First try the comprehensive knowledge base
        if KNOWLEDGE_BASE_AVAILABLE:
            knowledge_response = get_knowledge_for_query(message)
            if knowledge_response:
                return f"**Face Swapping Knowledge:** \n\n{knowledge_response}"
        
        # Fallback to basic knowledge
        message_lower = message.lower()
        
        for keyword, info in self.face_swap_knowledge.items():
            if keyword in message_lower:
                return f"**Face Swapping Help**: {info}"
        
        # Check for common face swapping terms
        if any(term in message_lower for term in ["face swap", "deepfake", "roop", "face replacement"]):
            welcome_msg = """**Welcome to Roop Unleashed AI Assistant!** 🤖

I can help you with:
- **Face swapping techniques** and best practices
- **Performance optimization** for faster processing
- **Troubleshooting** common issues
- **Video processing** tips and settings
- **Quality enhancement** options
- **GPU acceleration** setup

Try asking specific questions like:
- "How do I improve face swap quality?"
- "My processing is slow, how to optimize?"
- "What are the best settings for videos?"
- "How do I fix GPU memory errors?"
"""
            
            if KNOWLEDGE_BASE_AVAILABLE:
                tip = get_random_tip()
                welcome_msg += f"\n\n{tip}"
            
            return welcome_msg
        
        return None
    
    def _general_response(self, message: str) -> str:
        """Provide general responses for face swapping queries."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "help"]):
            return """👋 **Hello! I'm your Roop Unleashed AI Assistant.**

I'm here to help you with all aspects of face swapping and video processing. I have specialized knowledge about:

🎭 **Face Swapping**: Detection, quality, multiple faces
🎥 **Video Processing**: Codecs, quality, optimization  
⚡ **Performance**: GPU acceleration, memory optimization
🔧 **Troubleshooting**: Common errors and solutions
🎨 **Enhancement**: Face quality improvement tools

What would you like to know about face swapping?"""
        
        elif any(word in message_lower for word in ["quality", "improve", "better"]):
            return """🎨 **Improving Face Swap Quality:**

1. **Face Enhancement**: Enable GFPGAN or CodeFormer in Advanced settings
2. **Detection Settings**: Adjust face detection sensitivity for better matches
3. **Blend Ratio**: Fine-tune blending (0.7-0.8 usually works well)
4. **Input Quality**: Use high-resolution, well-lit source images
5. **Multiple Angles**: Select faces from similar angles/lighting
6. **Video Settings**: Use CRF 14-18 for high-quality video output

Try the 'optimization' agent for performance-specific advice!"""
        
        elif any(word in message_lower for word in ["slow", "fast", "speed", "performance"]):
            return """⚡ **Performance Optimization:**

1. **GPU Acceleration**: Enable CUDA/ROCm in Settings
2. **Memory Management**: Reduce frame buffer size if needed
3. **Threading**: Increase max threads (4-8 recommended)
4. **Video Resolution**: Process at lower resolution, upscale later
5. **Batch Processing**: Process multiple files together
6. **Close Apps**: Free up GPU memory by closing other applications

Use the 'optimization' agent for detailed performance analysis!"""
        
        elif any(word in message_lower for word in ["error", "problem", "issue", "crash"]):
            return """🔧 **Troubleshooting Common Issues:**

**GPU Errors**: Check memory usage, try CPU mode
**Model Loading**: Ensure models are downloaded properly
**Video Issues**: Check FFmpeg installation
**Out of Memory**: Reduce batch size or resolution
**Face Detection**: Adjust sensitivity or try different model

Use the 'troubleshooting' agent for specific error diagnosis!"""
        
        else:
            return """I'm your AI assistant for face swapping and video processing. I can help with:

• **Technical questions** about Roop Unleashed
• **Optimization tips** for better performance  
• **Quality improvement** techniques
• **Troubleshooting** errors and issues
• **Best practices** for face swapping

Try asking specific questions or use these specialized agents:
- `optimization` - Performance and speed
- `video` - Video processing help
- `troubleshooting` - Error resolution
- `rag` - Knowledge base search
- `auto` - Automatically route to best agent

What would you like to know?"""
    
    def clear_chat(self) -> Tuple[List, str]:
        """Clear the chat history."""
        self.conversation_history = []
        return [], ""
    
    def export_chat(self) -> str:
        """Export chat history as text."""
        if not self.conversation_history:
            return "No conversation to export."
        
        export_text = "# Roop Unleashed AI Chat Export\n\n"
        export_text += f"Exported on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i in range(0, len(self.conversation_history), 2):
            if i + 1 < len(self.conversation_history):
                user_msg = self.conversation_history[i].get('content', '')
                ai_msg = self.conversation_history[i + 1].get('content', '')
                
                export_text += f"## Conversation {(i // 2) + 1}\n\n"
                export_text += f"**User:** {user_msg}\n\n"
                export_text += f"**AI:** {ai_msg}\n\n"
                export_text += "---\n\n"
        
        return export_text
    
    def get_system_status(self) -> str:
        """Get system status information."""
        if self.agent_manager:
            status = self.agent_manager.get_system_status()
            status_text = "**System Status:**\n\n"
            status_text += f"• Available Agents: {status['agent_count']}\n"
            status_text += f"• Enhanced Features: {'✅' if status['enhanced_features'] else '❌'}\n"
            
            if 'llm_status' in status:
                llm_available = status['llm_status'].get('available', False)
                status_text += f"• LLM Integration: {'✅' if llm_available else '❌'}\n"
            
            if 'rag_status' in status:
                rag_docs = status['rag_status'].get('document_count', 0)
                status_text += f"• Knowledge Base: {rag_docs} documents\n"
            
            return status_text
        else:
            return "**System Status:** Basic mode (no agent manager)"


def create_chat_interface(settings=None) -> gr.Interface:
    """Create the AI Chat interface for Gradio."""
    
    chat_interface = AIChatInterface(settings)
    
    with gr.Column() as chat_component:
        gr.Markdown("## 🤖 AI Assistant Chat")
        gr.Markdown("Get help with face swapping, optimization, troubleshooting, and more!")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="AI Assistant",
                    height=400,
                    show_label=True,
                    avatar_images=(None, "🤖"),
                    type="messages"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything about face swapping, optimization, or troubleshooting...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    export_btn = gr.Button("Export Chat", variant="secondary")
                    help_btn = gr.Button("Quick Help", variant="secondary")
            
            with gr.Column(scale=1):
                agent_selector = gr.Dropdown(
                    choices=["auto"] + chat_interface.get_available_agents(),
                    value="auto",
                    label="AI Agent",
                    info="Select specialized agent or auto-route"
                )
                
                gr.Markdown("### 🎯 Specialized Agents:")
                gr.Markdown("""
                • **auto** - Smart routing to best agent
                • **optimization** - Performance & speed
                • **video** - Video processing help  
                • **troubleshooting** - Error resolution
                • **rag** - Knowledge base search
                • **operation** - General face-swapping
                """)
                
                # Quick help panel
                with gr.Accordion("📚 Quick Help Categories", open=False) as help_accordion:
                    if KNOWLEDGE_BASE_AVAILABLE:
                        for category, questions in HELP_CATEGORIES.items():
                            with gr.Accordion(f"📖 {category}", open=False):
                                for question in questions:
                                    help_question_btn = gr.Button(
                                        question, 
                                        variant="secondary", 
                                        size="sm",
                                        scale=1
                                    )
                                    # We'll wire these up later
                    else:
                        gr.Markdown("Knowledge base not available")
                
                system_status = gr.Textbox(
                    label="System Status",
                    value=chat_interface.get_system_status(),
                    lines=4,
                    interactive=False
                )
        
        # Chat export output
        export_output = gr.Textbox(
            label="Chat Export",
            lines=10,
            visible=False,
            interactive=False
        )
        
        # Event handlers
        def send_message(message, agent, history):
            return chat_interface.chat_response(message, agent, history)
        
        def show_export():
            export_text = chat_interface.export_chat()
            return gr.update(value=export_text, visible=True)
        
        def hide_export():
            return gr.update(visible=False)
        
        # Wire up events
        send_btn.click(
            send_message,
            inputs=[msg_input, agent_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            send_message,
            inputs=[msg_input, agent_selector, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            chat_interface.clear_chat,
            outputs=[chatbot, msg_input]
        )
        
        export_btn.click(
            show_export,
            outputs=[export_output]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[export_output]
        )
        
        # Hide export when clicking elsewhere
        msg_input.focus(hide_export, outputs=[export_output])
    
    return chat_component


# Example usage and testing
if __name__ == "__main__":
    # Test the chat interface
    interface = AIChatInterface()
    
    # Test responses
    test_queries = [
        "Hello, how can you help me?",
        "How do I improve face swap quality?",
        "My processing is very slow",
        "I'm getting GPU memory errors"
    ]
    
    for query in test_queries:
        response = interface._general_response(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")