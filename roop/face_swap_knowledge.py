"""
Face Swapping Knowledge Base for Roop Unleashed AI Assistant
Contains comprehensive information about face swapping techniques, best practices, and troubleshooting.
"""

FACE_SWAP_KNOWLEDGE_BASE = {
    "basics": {
        "what_is_face_swapping": """
Face swapping is the process of replacing one person's face with another's in images or videos using AI. 
Roop Unleashed uses advanced deep learning models like InsightFace for detection and custom neural networks for swapping.
        """,
        "how_it_works": """
1. **Face Detection**: Identifies faces in source and target media
2. **Feature Extraction**: Analyzes facial landmarks and features  
3. **Face Swapping**: Replaces target face with source face features
4. **Blending**: Seamlessly integrates the swapped face
5. **Enhancement**: Optional post-processing for quality improvement
        """,
        "input_requirements": """
**Source Image**: Clear, well-lit face photo (preferably front-facing)
**Target Media**: Images or videos with visible faces
**Quality**: Higher resolution inputs generally produce better results
**Lighting**: Similar lighting conditions between source and target help
        """
    },
    
    "quality_tips": {
        "best_practices": """
🎯 **For Best Quality Results:**

1. **Source Image Selection**:
   - Use high-resolution images (1024px+ recommended)
   - Ensure clear, well-lit facial features
   - Front-facing or similar angle to target
   - Minimal shadows and even lighting

2. **Target Selection**:
   - Similar head pose and angle
   - Good lighting conditions
   - Clear facial visibility
   - Avoid extreme expressions or occlusions

3. **Settings Optimization**:
   - Face detection sensitivity: 0.6-0.8
   - Blend ratio: 0.7-0.8 for natural look
   - Enable face enhancement (GFPGAN/CodeFormer)
   - Use appropriate video quality (CRF 14-20)
        """,
        "enhancement_options": """
**Face Enhancers Available:**

• **GFPGAN**: General face enhancement, good for overall quality
• **CodeFormer**: Better for heavily compressed or low-quality faces
• **DMDNet**: Specialized for specific enhancement scenarios

**When to Use Enhancement:**
- Low-quality source images
- Visible artifacts after swapping
- Blurry or pixelated results
- Video compression artifacts
        """,
        "common_issues": """
**Quality Issues & Solutions:**

❌ **Blurry faces**: Increase source resolution, enable enhancement
❌ **Color mismatch**: Adjust blend ratio, check lighting
❌ **Visible edges**: Fine-tune detection sensitivity
❌ **Flickering in video**: Enable temporal consistency
❌ **Unnatural look**: Try different face detection models
        """
    },
    
    "performance": {
        "gpu_acceleration": """
🚀 **GPU Acceleration Setup:**

**NVIDIA (CUDA):**
- Install CUDA 11.8 or 12.1
- Set provider to 'CUDAExecutionProvider'
- Monitor GPU memory usage

**AMD (ROCm):**
- Install ROCm drivers
- Set provider to 'ROCMExecutionProvider'
- Linux support primarily

**Intel (DirectML):**
- Windows 10/11 with DirectX 12
- Set provider to 'DmlExecutionProvider'
- Good for integrated graphics

**Apple (MPS):**
- macOS with Apple Silicon
- Set provider to 'CoreMLExecutionProvider'
- M1/M2 Mac optimization
        """,
        "optimization_settings": """
⚡ **Performance Optimization:**

**Memory Management:**
- Frame buffer size: 2-4 (reduce if out of memory)
- Max threads: 4-8 (match CPU cores)
- Memory limit: Set based on available RAM

**Processing Speed:**
- Use GPU acceleration when available
- Lower video resolution for faster processing
- Batch process multiple files
- Close unnecessary applications

**Quality vs Speed:**
- Lower CRF values = higher quality, slower speed
- Face detection sensitivity affects speed
- Enhancement adds processing time
        """,
        "troubleshooting_performance": """
🔧 **Performance Issues:**

**Slow Processing:**
1. Enable GPU acceleration
2. Increase max threads (CPU)
3. Reduce video resolution temporarily
4. Disable unnecessary enhancements

**Out of Memory:**
1. Reduce frame buffer size
2. Lower batch size
3. Use CPU mode as fallback
4. Close other applications

**GPU Not Detected:**
1. Verify driver installation
2. Check CUDA/ROCm compatibility
3. Restart application after driver update
4. Monitor GPU usage with system tools
        """
    },
    
    "video_processing": {
        "supported_formats": """
📹 **Video Format Support:**

**Input Formats:**
- MP4 (recommended)
- AVI, MOV, MKV
- WebM, FLV
- Most FFmpeg-supported formats

**Output Options:**
- Codec: H.264 (default), H.265, VP9
- Container: MP4, AVI, MKV
- Quality: CRF 0-51 (14-20 recommended)

**Audio Handling:**
- Keep original audio (default)
- Skip audio option available
- Audio sync preserved
        """,
        "video_settings": """
🎬 **Optimal Video Settings:**

**Quality Balance:**
- CRF 14-18: High quality, larger files
- CRF 20-23: Good quality, moderate size
- CRF 24-28: Lower quality, smaller files

**Frame Rate:**
- Keep original FPS (recommended)
- 24-30 FPS for most content
- 60 FPS for smooth motion

**Resolution:**
- Process at original resolution when possible
- 1080p recommended for quality
- 720p for faster processing
        """,
        "batch_processing": """
⚙️ **Batch Video Processing:**

**Setup:**
1. Select multiple target files
2. Configure settings once
3. Enable batch processing
4. Monitor progress

**Best Practices:**
- Process similar videos together
- Use consistent settings
- Monitor system resources
- Save settings profile for reuse

**Tips:**
- Start with shorter clips for testing
- Use same source face for consistency
- Check first result before batch processing
        """
    },
    
    "troubleshooting": {
        "common_errors": """
🚨 **Common Errors & Solutions:**

**"CUDA out of memory":**
- Reduce frame buffer size
- Lower video resolution
- Close other GPU applications
- Switch to CPU mode temporarily

**"No faces detected":**
- Adjust face detection sensitivity
- Try different detection model
- Ensure faces are clearly visible
- Check image/video quality

**"Model loading failed":**
- Check internet connection
- Verify model download completion
- Clear model cache and re-download
- Check disk space

**"FFmpeg error":**
- Install/update FFmpeg
- Check video file integrity
- Try different input format
- Verify codec support
        """,
        "installation_issues": """
🔧 **Installation Troubleshooting:**

**Dependencies:**
- Python 3.9-3.12 required
- Visual C++ redistributables (Windows)
- CUDA/ROCm drivers for GPU
- FFmpeg for video processing

**Common Fixes:**
1. Update pip: `pip install --upgrade pip`
2. Use virtual environment
3. Install Visual Studio Build Tools (Windows)
4. Check antivirus software interference
5. Run as administrator if needed

**Environment Setup:**
- Use conda/virtualenv for isolation
- Verify CUDA version compatibility
- Check PATH environment variables
- Install dependencies in correct order
        """,
        "model_issues": """
📦 **Model Download/Loading Issues:**

**Download Problems:**
- Check internet connectivity
- Verify available disk space (2GB+ needed)
- Try manual model download
- Clear temporary files

**Loading Errors:**
- Verify model file integrity
- Check file permissions
- Restart application
- Clear model cache

**Model Locations:**
- Models stored in ./models/ directory
- Check for corrupted downloads
- Re-download if necessary
- Verify checksums if available
        """
    },
    
    "advanced": {
        "multiple_faces": """
👥 **Multiple Face Swapping:**

**Detection:**
- Roop detects multiple faces automatically
- Select specific faces from gallery
- Use face selection mode for precision

**Swapping Modes:**
- First detected face
- Selected face index
- Gender-based selection
- Manual face selection

**Best Practices:**
- Use high-quality source images for each face
- Maintain consistent lighting
- Process faces individually for better control
- Preview results before full processing
        """,
        "masking_features": """
🎭 **Face Masking & Occlusion:**

**Text-based Masking:**
- Use natural language prompts
- Example: "glasses", "hat", "hands"
- Helps avoid swapping occluded areas

**Manual Masking:**
- Draw custom masks if needed
- Exclude unwanted areas
- Fine-tune boundaries

**Occlusion Handling:**
- Automatic detection of obstacles
- Smart blending around occlusions
- Preserve original content where appropriate
        """,
        "temporal_consistency": """
🎬 **Temporal Consistency (Video):**

**What it does:**
- Reduces flickering between frames
- Maintains consistent face tracking
- Improves video quality

**Settings:**
- Enable in Advanced options
- May increase processing time
- Especially important for longer videos

**When to use:**
- Professional video work
- Noticeable frame inconsistencies
- High-quality output requirements
        """
    },
    
    "ethics_and_legal": {
        "responsible_use": """
⚖️ **Ethical Guidelines:**

**Consent & Permission:**
- Always get consent before using someone's likeness
- Respect privacy and personal rights
- Be transparent about AI-generated content

**Disclosure:**
- Clearly label deepfake content
- Don't misrepresent as authentic
- Follow platform guidelines

**Prohibited Uses:**
- Non-consensual content
- Harassment or impersonation
- Misinformation or fraud
- Illegal activities
        """,
        "legal_considerations": """
📜 **Legal Considerations:**

**Copyright & Rights:**
- Respect image rights and licenses
- Don't use copyrighted material without permission
- Consider personality rights

**Platform Policies:**
- Check social media guidelines
- Respect content policies
- Follow community standards

**Jurisdiction:**
- Laws vary by location
- Consult legal advice when uncertain
- Stay informed about changing regulations
        """
    }
}


def get_knowledge_for_query(query: str) -> str:
    """
    Search the knowledge base for relevant information based on the query.
    """
    query_lower = query.lower()
    
    # Map keywords to knowledge sections with more specific matching
    keyword_mapping = {
        # Basics
        ("what is", "how does", "explain", "basics", "introduction", "start"): "basics",
        
        # Quality
        ("quality", "improve", "better", "enhance", "clear", "sharp", "blurry", "artifact"): "quality_tips",
        
        # Performance  
        ("slow", "fast", "speed", "performance", "optimize", "gpu", "cuda", "memory", "ram"): "performance",
        
        # Video
        ("video", "mp4", "avi", "frame", "fps", "codec", "ffmpeg", "mov", "mkv"): "video_processing",
        
        # Troubleshooting
        ("error", "problem", "issue", "crash", "fail", "not working", "broken", "bug"): "troubleshooting",
        
        # Advanced
        ("multiple", "mask", "temporal", "advanced", "professional", "batch"): "advanced",
        
        # Ethics
        ("legal", "ethical", "consent", "responsible", "permission"): "ethics_and_legal"
    }
    
    # Find relevant sections
    relevant_sections = []
    for keywords, section in keyword_mapping.items():
        if any(keyword in query_lower for keyword in keywords):
            relevant_sections.append(section)
    
    # Search for specific subsection matches
    all_responses = []
    
    for section_name in relevant_sections or FACE_SWAP_KNOWLEDGE_BASE.keys():
        section_data = FACE_SWAP_KNOWLEDGE_BASE.get(section_name, {})
        
        for subsection_key, content in section_data.items():
            # Check if query matches subsection keywords
            subsection_keywords = subsection_key.replace("_", " ").split()
            if any(keyword in query_lower for keyword in subsection_keywords):
                all_responses.append(content.strip())
            # Also check content for query terms
            elif any(term in content.lower() for term in query_lower.split() if len(term) > 3):
                all_responses.append(content.strip())
    
    # Remove duplicates and limit response length
    unique_responses = list(dict.fromkeys(all_responses))  # Preserves order
    
    if unique_responses:
        # Return first 2 most relevant responses to avoid overwhelming
        return "\n\n".join(unique_responses[:2])
    
    return ""


def get_random_tip() -> str:
    """Get a random helpful tip about face swapping."""
    tips = [
        "💡 **Tip**: Use high-resolution source images for better face swap quality!",
        "⚡ **Tip**: Enable GPU acceleration in Settings for faster processing!",
        "🎯 **Tip**: Similar lighting between source and target faces gives more natural results!",
        "🔧 **Tip**: Adjust blend ratio to 0.7-0.8 for most natural-looking swaps!",
        "📹 **Tip**: Use CRF 14-20 for good quality vs file size balance in videos!",
        "🎭 **Tip**: Face enhancement can significantly improve low-quality sources!",
        "⚙️ **Tip**: Batch processing multiple files saves time with consistent settings!",
        "🚀 **Tip**: Close other GPU-using applications to free memory for processing!"
    ]
    
    import random
    return random.choice(tips)


# Categories for organized help
HELP_CATEGORIES = {
    "Getting Started": [
        "What is face swapping?",
        "How does face swapping work?",
        "What do I need to get started?",
        "Best practices for beginners"
    ],
    "Quality & Enhancement": [
        "How to improve face swap quality?",
        "Face enhancement options",
        "Dealing with blurry results",
        "Color matching tips"
    ],
    "Performance": [
        "GPU acceleration setup",
        "Optimization settings",
        "Memory management",
        "Speed vs quality balance"
    ],
    "Video Processing": [
        "Supported video formats",
        "Video quality settings",
        "Batch processing videos",
        "Audio handling"
    ],
    "Troubleshooting": [
        "Common error messages",
        "Installation problems",
        "Model loading issues",
        "GPU detection problems"
    ],
    "Advanced Features": [
        "Multiple face swapping",
        "Face masking techniques",
        "Temporal consistency",
        "Professional workflows"
    ]
}


if __name__ == "__main__":
    # Test the knowledge system
    test_queries = [
        "How do I improve face swap quality?",
        "GPU memory error",
        "Video processing settings",
        "Multiple faces in one image"
    ]
    
    for query in test_queries:
        result = get_knowledge_for_query(query)
        print(f"Query: {query}")
        print(f"Response: {result[:200]}...")
        print("-" * 50)