# Enhanced Face Processing Documentation

This document describes the enhanced face recognition and blending capabilities added to roop-unleashed.

## Overview

The enhanced face processing system improves face recognition accuracy, blending quality, and overall output realism through:

- **Advanced Face Detection**: Multi-scale detection with quality assessment
- **Sophisticated Blending**: Poisson blending, multi-band blending, and gradient-based methods
- **Quality Metrics**: Automated quality assessment and validation
- **Adaptive Processing**: Dynamic parameter adjustment based on image characteristics

## Features

### 1. Enhanced Face Detection (`roop/enhanced_face_detection.py`)

#### Face Quality Assessment
- **Detection confidence**: Face detection reliability score
- **Size optimization**: Face size relative to frame for optimal processing
- **Pose analysis**: Face orientation and symmetry assessment
- **Sharpness detection**: Image clarity using Laplacian variance
- **Lighting evaluation**: Histogram-based lighting quality

#### Adaptive Detection
- **Resolution-based sizing**: Optimal detection parameters for different image resolutions
- **Multi-scale detection**: Processes images at multiple scales for better small face detection
- **Non-maximum suppression**: Removes duplicate detections

#### Usage Example
```python
from roop.enhanced_face_detection import get_enhanced_faces, get_best_face

# Get faces with quality scores
faces_with_quality = get_enhanced_faces(frame, quality_threshold=0.5)

# Get the highest quality face
best_face = get_best_face(frame)
```

### 2. Advanced Blending (`roop/advanced_blending.py`)

#### Blending Methods

**Poisson Blending**
- Seamless integration using gradient-domain processing
- Solves Poisson equation for natural blending
- Best for: Complex lighting conditions, detailed textures

**Multi-band Blending**
- Laplacian pyramid-based blending
- Preserves different frequency components
- Best for: Color matching, texture preservation

**Gradient Blending**
- Edge-preserving gradient-based method
- Maintains image structure while blending
- Best for: Sharp features, edge preservation

**Alpha Blending**
- Traditional weighted blending with improvements
- Enhanced edge smoothing and masking
- Best for: Quick processing, simple scenarios

#### Usage Example
```python
from roop.advanced_blending import AdvancedBlender

blender = AdvancedBlender()
result = blender.blend_face(
    source_face, target_frame, face_bbox,
    blend_method="multiband",  # or "poisson", "gradient", "alpha"
    blend_ratio=0.8
)
```

### 3. Enhanced Face Swapper (`roop/enhanced_face_swapper.py`)

#### Features
- **Quality-based face selection**: Automatically selects best faces for swapping
- **Preprocessing**: Face normalization and enhancement before swapping
- **Quality validation**: Assesses swap result quality
- **Multiple processing modes**: Best quality, all faces, or target matching

#### Face Selection Modes
- `best_quality`: Swap only the highest quality detected face
- `all_faces`: Swap all faces above quality threshold
- `match_target`: Match faces based on embedding similarity

#### Usage Example
```python
from roop.enhanced_face_swapper import enhanced_process_frame

result = enhanced_process_frame(
    source_face, target_face, frame,
    face_selection_mode="best_quality",
    blend_method="multiband",
    blend_ratio=0.8,
    min_quality=0.4
)
```

## Configuration

### Global Settings (`roop/globals.py`)

New configuration options:

```python
# Enhanced processing settings
use_enhanced_processing = False    # Enable/disable enhanced processing
blend_method = "multiband"         # Default blending method
quality_threshold = 0.4            # Minimum face quality threshold
adaptive_detection = True          # Enable adaptive detection sizing
```

### Runtime Configuration

```python
from roop.processors.frame.face_swapper import enable_enhanced_processing, set_blend_method

# Enable enhanced processing
enable_enhanced_processing(True)

# Set blending method
set_blend_method("multiband")

# Get available methods
methods = get_blend_methods()
print(f"Available methods: {methods}")
```

## Performance Considerations

### Speed vs Quality Trade-offs

**Standard Processing**: Fastest, good for real-time applications
**Enhanced Processing**: Higher quality but slower, best for final output

### Optimization Tips

1. **Adaptive Detection**: Automatically adjusts detection resolution
2. **Quality Thresholds**: Higher thresholds skip low-quality faces
3. **Blending Method Selection**:
   - `alpha`: Fastest
   - `multiband`: Good balance
   - `gradient`: High quality
   - `poisson`: Highest quality, slowest

### Memory Usage

Enhanced processing uses more memory due to:
- Multi-scale detection
- Pyramid-based blending
- Quality assessment buffers

## Quality Assessment

### Frame Quality Metrics

```python
from roop.enhanced_face_swapper import assess_frame_quality

quality_info = assess_frame_quality(frame)
# Returns:
# {
#     "overall_quality": 0.85,
#     "face_count": 2,
#     "best_face_quality": 0.85,
#     "average_face_quality": 0.72
# }
```

### Swap Quality Assessment

The system automatically assesses swap quality based on:
- Visual artifact detection
- Color consistency with surroundings
- Edge quality and smoothness

## Integration with Existing System

The enhanced system maintains full backward compatibility:

- **Automatic fallback**: If enhanced processing fails, falls back to standard
- **Configuration-based**: Enable/disable via settings
- **Transparent integration**: Works with existing UI and processing pipelines

## Best Practices

### For Best Results

1. **Use appropriate quality thresholds**: Start with 0.4, adjust based on content
2. **Choose blending method based on content**:
   - Portraits: `multiband` or `poisson`
   - Action scenes: `gradient` or `alpha`
   - Batch processing: `alpha` for speed
3. **Enable adaptive detection**: Let the system optimize parameters
4. **Monitor quality metrics**: Use assessment functions to validate results

### Troubleshooting

**Low quality results**: Lower quality threshold or try different blend methods
**Slow processing**: Use `alpha` blending or increase quality threshold
**Memory issues**: Disable multi-scale detection or reduce batch size

## Future Enhancements

Planned improvements:
- Real-time optimization
- GPU acceleration for blending
- Machine learning-based quality prediction
- Advanced face alignment techniques
- Temporal consistency for video processing

## API Reference

### Key Functions

```python
# Enhanced face detection
get_enhanced_faces(frame, quality_threshold=0.5)
get_best_face(frame)
FaceQualityAssessment.calculate_face_quality(face, frame)

# Advanced blending
AdvancedBlender.blend_face(source, target, bbox, method, ratio)
get_available_blend_methods()

# Enhanced processing
enhanced_process_frame(source_face, target_face, frame, **kwargs)
assess_frame_quality(frame)
get_enhancement_config()

# Configuration
enable_enhanced_processing(enable=True)
set_blend_method(method)
get_processing_info()
```

### Parameters

**quality_threshold**: Float 0.0-1.0, minimum face quality to process
**blend_method**: String, one of ["alpha", "multiband", "gradient", "poisson"]
**blend_ratio**: Float 0.0-1.0, blending strength
**face_selection_mode**: String, one of ["best_quality", "all_faces", "match_target"]

## Examples

See `test_enhancements.py` and `test_structure.py` for usage examples and testing procedures.