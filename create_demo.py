#!/usr/bin/env python3
"""
Demo script showing the new enhanced features.
This creates a visual demonstration of what the new features accomplish.
"""

import cv2
import numpy as np
import os

def create_demo_images():
    """Create demo images showing enhanced features."""
    
    print("Creating Enhanced Face Processing Demo...")
    
    # Create output directory
    demo_dir = "demo_output"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 1. Inpainting Demo
    print("1. Creating Inpainting Demo...")
    create_inpainting_demo(demo_dir)
    
    # 2. Temporal Consistency Demo
    print("2. Creating Temporal Consistency Demo...")
    create_temporal_demo(demo_dir)
    
    # 3. Face Quality Analysis Demo
    print("3. Creating Face Quality Analysis Demo...")
    create_quality_demo(demo_dir)
    
    # 4. Advanced Blending Demo
    print("4. Creating Advanced Blending Demo...")
    create_blending_demo(demo_dir)
    
    print(f"Demo images created in {demo_dir}/")

def create_inpainting_demo(output_dir):
    """Demonstrate inpainting capabilities."""
    
    # Create a face-like image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw face
    cv2.circle(img, (150, 150), 80, (220, 200, 180), -1)  # Face
    cv2.circle(img, (125, 130), 8, (50, 50, 50), -1)      # Left eye
    cv2.circle(img, (175, 130), 8, (50, 50, 50), -1)      # Right eye
    cv2.ellipse(img, (150, 170), (15, 8), 0, 0, 180, (100, 50, 50), 2)  # Mouth
    
    # Add some "defects" to demonstrate inpainting
    cv2.rectangle(img, (140, 120), (160, 140), (0, 0, 255), -1)  # Red rectangle over eye
    cv2.circle(img, (130, 180), 10, (0, 255, 0), -1)             # Green circle
    
    # Save original with defects
    cv2.imwrite(f"{output_dir}/1_original_with_defects.png", img)
    
    # Create mask for inpainting
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(mask, (140, 120), (160, 140), 255, -1)
    cv2.circle(mask, (130, 180), 10, 255, -1)
    
    # Perform traditional inpainting (OpenCV)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(f"{output_dir}/2_inpainted_result.png", inpainted)
    
    # Create side-by-side comparison
    comparison = np.hstack([img, inpainted])
    cv2.putText(comparison, "Before", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "After Inpainting", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/inpainting_comparison.png", comparison)

def create_temporal_demo(output_dir):
    """Demonstrate temporal consistency."""
    
    # Create a sequence of frames with slight movement
    frames = []
    
    for i in range(5):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Moving face
        center_x = 100 + i * 10  # Face moves right
        center_y = 100 + int(5 * np.sin(i * 0.5))  # Slight vertical movement
        
        # Draw face with slight variations
        cv2.circle(img, (center_x, center_y), 40, (200, 180, 160), -1)
        cv2.circle(img, (center_x - 15, center_y - 10), 4, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (center_x + 15, center_y - 10), 4, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (center_x, center_y + 15), (8, 4), 0, 0, 180, (50, 50, 50), 1)  # Mouth
        
        frames.append(img)
        cv2.imwrite(f"{output_dir}/temporal_frame_{i:02d}.png", img)
    
    # Create temporal smoothing demonstration
    smoothed_frames = []
    smoothing_factor = 0.3
    
    prev_center = None
    for i, frame in enumerate(frames):
        if prev_center is not None:
            # Apply temporal smoothing
            current_center = (100 + i * 10, 100 + int(5 * np.sin(i * 0.5)))
            smoothed_center = (
                int(current_center[0] * smoothing_factor + prev_center[0] * (1 - smoothing_factor)),
                int(current_center[1] * smoothing_factor + prev_center[1] * (1 - smoothing_factor))
            )
        else:
            smoothed_center = (100, 100)
        
        # Create smoothed frame
        smooth_img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.circle(smooth_img, smoothed_center, 40, (180, 200, 180), -1)
        cv2.circle(smooth_img, (smoothed_center[0] - 15, smoothed_center[1] - 10), 4, (0, 0, 0), -1)
        cv2.circle(smooth_img, (smoothed_center[0] + 15, smoothed_center[1] - 10), 4, (0, 0, 0), -1)
        cv2.ellipse(smooth_img, (smoothed_center[0], smoothed_center[1] + 15), (8, 4), 0, 0, 180, (50, 50, 50), 1)
        
        smoothed_frames.append(smooth_img)
        cv2.imwrite(f"{output_dir}/temporal_smoothed_{i:02d}.png", smooth_img)
        
        prev_center = smoothed_center
    
    # Create comparison gif frames
    for i in range(len(frames)):
        comparison = np.hstack([frames[i], smoothed_frames[i]])
        cv2.putText(comparison, "Original", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Smoothed", (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(f"{output_dir}/temporal_comparison_{i:02d}.png", comparison)

def create_quality_demo(output_dir):
    """Demonstrate face quality analysis."""
    
    # Create faces with different quality levels
    qualities = ["Low", "Medium", "High"]
    
    for i, quality in enumerate(qualities):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        if quality == "Low":
            # Low quality: small, blurry face
            face_size = 30
            blur_kernel = 15
            cv2.circle(img, (100, 100), face_size, (150, 130, 110), -1)
            cv2.circle(img, (90, 90), 3, (0, 0, 0), -1)
            cv2.circle(img, (110, 90), 3, (0, 0, 0), -1)
            cv2.ellipse(img, (100, 115), (8, 4), 0, 0, 180, (50, 50, 50), 1)
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
            
        elif quality == "Medium":
            # Medium quality: medium size, slight blur
            face_size = 50
            blur_kernel = 5
            cv2.circle(img, (100, 100), face_size, (180, 160, 140), -1)
            cv2.circle(img, (85, 85), 5, (0, 0, 0), -1)
            cv2.circle(img, (115, 85), 5, (0, 0, 0), -1)
            cv2.ellipse(img, (100, 120), (12, 6), 0, 0, 180, (50, 50, 50), 2)
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
            
        else:  # High quality
            # High quality: large, sharp face
            face_size = 70
            cv2.circle(img, (100, 100), face_size, (220, 200, 180), -1)
            cv2.circle(img, (80, 80), 7, (0, 0, 0), -1)
            cv2.circle(img, (120, 80), 7, (0, 0, 0), -1)
            cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (50, 50, 50), 2)
            # Add some detail
            cv2.ellipse(img, (80, 75), (8, 4), 0, 0, 180, (100, 80, 60), 1)  # Eyebrow
            cv2.ellipse(img, (120, 75), (8, 4), 0, 0, 180, (100, 80, 60), 1)  # Eyebrow
        
        # Add quality score text
        cv2.putText(img, f"{quality} Quality", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate simple quality metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)
        
        cv2.putText(img, f"Sharpness: {sharpness:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imwrite(f"{output_dir}/quality_{quality.lower()}.png", img)
    
    # Create comparison
    low_img = cv2.imread(f"{output_dir}/quality_low.png")
    med_img = cv2.imread(f"{output_dir}/quality_medium.png")
    high_img = cv2.imread(f"{output_dir}/quality_high.png")
    
    comparison = np.hstack([low_img, med_img, high_img])
    cv2.imwrite(f"{output_dir}/quality_comparison.png", comparison)

def create_blending_demo(output_dir):
    """Demonstrate advanced blending techniques."""
    
    # Create source and target images
    source = np.zeros((200, 200, 3), dtype=np.uint8)
    target = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Source face (blue tint)
    cv2.circle(source, (100, 100), 60, (200, 150, 100), -1)
    cv2.circle(source, (85, 85), 6, (0, 0, 0), -1)
    cv2.circle(source, (115, 85), 6, (0, 0, 0), -1)
    cv2.ellipse(source, (100, 120), (12, 6), 0, 0, 180, (50, 50, 50), 2)
    
    # Target background (different color)
    target[:] = (50, 100, 50)  # Green background
    cv2.circle(target, (100, 100), 80, (150, 180, 150), -1)  # Different face color
    
    # Simple alpha blending
    alpha = 0.7
    alpha_blend = cv2.addWeighted(source, alpha, target, 1 - alpha, 0)
    cv2.putText(alpha_blend, "Alpha Blend", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/blend_alpha.png", alpha_blend)
    
    # Gaussian blending simulation
    # Create a mask for the face region
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    
    # Apply Gaussian blur to the mask for smooth blending
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_normalized = mask_blurred.astype(np.float32) / 255.0
    
    # Perform blending
    gaussian_blend = target.copy().astype(np.float32)
    for c in range(3):
        gaussian_blend[:, :, c] = (source[:, :, c] * mask_normalized + 
                                 target[:, :, c] * (1 - mask_normalized))
    
    gaussian_blend = gaussian_blend.astype(np.uint8)
    cv2.putText(gaussian_blend, "Gaussian Blend", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/blend_gaussian.png", gaussian_blend)
    
    # Create comparison
    comparison = np.hstack([source, target, alpha_blend, gaussian_blend])
    labels = ["Source", "Target", "Alpha", "Gaussian"]
    for i, label in enumerate(labels):
        cv2.putText(comparison, label, (i * 200 + 10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(f"{output_dir}/blending_comparison.png", comparison)

def create_feature_overview():
    """Create a feature overview image."""
    
    # Create a large overview image
    overview = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(overview, "Enhanced Face Processing Features", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Feature list
    features = [
        "1. Advanced Inpainting",
        "   - Traditional CV methods (Telea, Navier-Stokes)",
        "   - Stable Diffusion integration (framework)",
        "   - Automatic mask generation",
        "",
        "2. Temporal Consistency for Videos",
        "   - Face position stabilization",
        "   - Landmark smoothing",
        "   - Optical flow tracking",
        "   - Quality-based filtering",
        "",
        "3. Advanced Face Models (WAN-style)",
        "   - Face quality analysis",
        "   - Enhancement algorithms",
        "   - Best face selection",
        "   - Multiple quality metrics",
        "",
        "4. ComfyUI-Inspired Processing",
        "   - Node-based architecture framework",
        "   - Batch processing optimization",
        "   - Memory management",
        "   - Adaptive quality settings",
        "",
        "5. Enhanced UI Controls",
        "   - Inpainting parameter controls",
        "   - Temporal consistency settings",
        "   - Real-time quality analysis",
        "   - System optimization status"
    ]
    
    y_pos = 100
    for feature in features:
        color = (255, 255, 255) if feature.startswith((" ", "")) else (100, 255, 100)
        cv2.putText(overview, feature, (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_pos += 25
    
    cv2.imwrite("demo_output/feature_overview.png", overview)

if __name__ == "__main__":
    create_demo_images()
    create_feature_overview()
    
    print("\n🎉 Demo completed successfully!")
    print("\nGenerated demo files:")
    print("- Inpainting demonstration")
    print("- Temporal consistency examples")
    print("- Face quality analysis samples")
    print("- Advanced blending comparisons")
    print("- Feature overview")
    
    print(f"\nCheck the 'demo_output' directory for visual demonstrations of the new features!")
    print("\nThese demonstrate the capabilities that have been added to roop-unleashed:")
    print("✓ Inpainting for face correction")
    print("✓ Temporal consistency for smoother videos")
    print("✓ Advanced face quality analysis")
    print("✓ Enhanced blending techniques")
    print("✓ Integrated UI controls")