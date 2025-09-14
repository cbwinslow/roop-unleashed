"""
Enhanced configuration system for advanced face processing features.
Provides centralized configuration for all enhancement modules.
"""

from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Processing priority levels."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class InterpolationMethod(Enum):
    """Frame interpolation methods."""
    RIFE = "rife"
    ADAPTIVE = "adaptive"
    SIMPLE = "simple"


class InpaintingMethod(Enum):
    """Inpainting methods."""
    TRADITIONAL_TELEA = "traditional_telea"
    TRADITIONAL_NS = "traditional_ns"
    EDGE_AWARE = "edge_aware"
    CONTEXT_AWARE = "context_aware"
    STABLE_DIFFUSION = "stable_diffusion"


class FaceEnhancementModel(Enum):
    """Face enhancement models."""
    REAL_ESRGAN = "real_esrgan"
    RESTORE_FORMER = "restore_former"
    WAN_ENHANCEMENT = "wan_enhancement"
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"


@dataclass
class VideoEnhancementConfig:
    """Configuration for video enhancement features."""
    
    # Frame rate enhancement
    enable_frame_interpolation: bool = False
    target_fps: Optional[float] = None
    interpolation_method: InterpolationMethod = InterpolationMethod.ADAPTIVE
    max_interpolation_factor: int = 4
    
    # Temporal consistency
    enable_temporal_consistency: bool = True
    temporal_smoothing_factor: float = 0.3
    enable_optical_flow: bool = True
    flow_quality: str = "medium"  # low, medium, high
    
    # Quality settings
    maintain_aspect_ratio: bool = True
    enable_quality_optimization: bool = True
    adaptive_quality: bool = True


@dataclass
class FaceEnhancementConfig:
    """Configuration for face enhancement features."""
    
    # Model selection
    primary_model: FaceEnhancementModel = FaceEnhancementModel.REAL_ESRGAN
    fallback_model: FaceEnhancementModel = FaceEnhancementModel.GFPGAN
    
    # Enhancement parameters
    enhancement_level: float = 0.8  # 0.0 to 1.0
    upscale_factor: int = 2  # 1, 2, 4, 8
    
    # Quality thresholds
    minimum_face_quality: float = 0.4
    target_face_quality: float = 0.8
    
    # Processing options
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    preserve_original_lighting: bool = True
    
    # Real-ESRGAN specific
    realesrgan_tile_size: int = 512
    realesrgan_scale: int = 4
    
    # RestoreFormer specific
    restoreformer_restoration_level: float = 0.8
    
    # WAN model specific
    wan_model_type: str = "enhancement"


@dataclass
class InpaintingConfig:
    """Configuration for inpainting and obstruction correction."""
    
    # Method selection
    primary_method: InpaintingMethod = InpaintingMethod.EDGE_AWARE
    fallback_method: InpaintingMethod = InpaintingMethod.TRADITIONAL_TELEA
    
    # Detection settings
    enable_smart_detection: bool = True
    enable_glasses_detection: bool = True
    enable_hair_detection: bool = True
    enable_mask_detection: bool = True
    
    # Mask generation
    boundary_expand_ratio: float = 0.1
    mask_feather_radius: int = 5
    
    # Processing parameters
    inpaint_radius: int = 3
    enable_edge_preservation: bool = True
    context_search_radius: int = 20
    
    # Quality settings
    enable_multi_pass: bool = False
    max_iterations: int = 3


@dataclass
class FaceProfileConfig:
    """Configuration for enhanced face profiling."""
    
    # Quality assessment
    enable_quality_analysis: bool = True
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpness": 0.25,
        "lighting": 0.20,
        "resolution": 0.20,
        "pose": 0.25,
        "artifacts": 0.10
    })
    
    # Selection criteria
    minimum_quality_score: float = 0.6
    prefer_frontal_faces: bool = True
    max_pose_angle: float = 30.0  # degrees
    
    # Profile generation
    enable_multi_angle_profiling: bool = True
    generate_face_variations: bool = True
    max_variations: int = 5
    
    # Pose estimation
    enable_pose_estimation: bool = True
    pose_estimation_method: str = "pnp"  # pnp, dnn, mediapipe


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Processing priority
    priority: ProcessingPriority = ProcessingPriority.BALANCED
    
    # Memory management
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    enable_garbage_collection: bool = True
    
    # GPU settings
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = False
    
    # Threading
    max_worker_threads: int = 4
    enable_parallel_processing: bool = True
    
    # Batch processing
    batch_size: int = 1
    enable_batch_optimization: bool = False


@dataclass
class EnhancedProcessingConfig:
    """Main configuration class for all enhanced processing features."""
    
    # Sub-configurations
    video: VideoEnhancementConfig = field(default_factory=VideoEnhancementConfig)
    face_enhancement: FaceEnhancementConfig = field(default_factory=FaceEnhancementConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    face_profile: FaceProfileConfig = field(default_factory=FaceProfileConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global settings
    enable_enhanced_features: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Compatibility
    maintain_backwards_compatibility: bool = True
    enable_fallback_methods: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "video": {
                "enable_frame_interpolation": self.video.enable_frame_interpolation,
                "target_fps": self.video.target_fps,
                "interpolation_method": self.video.interpolation_method.value,
                "max_interpolation_factor": self.video.max_interpolation_factor,
                "enable_temporal_consistency": self.video.enable_temporal_consistency,
                "temporal_smoothing_factor": self.video.temporal_smoothing_factor,
                "enable_optical_flow": self.video.enable_optical_flow,
                "flow_quality": self.video.flow_quality,
                "maintain_aspect_ratio": self.video.maintain_aspect_ratio,
                "enable_quality_optimization": self.video.enable_quality_optimization,
                "adaptive_quality": self.video.adaptive_quality
            },
            "face_enhancement": {
                "primary_model": self.face_enhancement.primary_model.value,
                "fallback_model": self.face_enhancement.fallback_model.value,
                "enhancement_level": self.face_enhancement.enhancement_level,
                "upscale_factor": self.face_enhancement.upscale_factor,
                "minimum_face_quality": self.face_enhancement.minimum_face_quality,
                "target_face_quality": self.face_enhancement.target_face_quality,
                "enable_preprocessing": self.face_enhancement.enable_preprocessing,
                "enable_postprocessing": self.face_enhancement.enable_postprocessing,
                "preserve_original_lighting": self.face_enhancement.preserve_original_lighting,
                "realesrgan_tile_size": self.face_enhancement.realesrgan_tile_size,
                "realesrgan_scale": self.face_enhancement.realesrgan_scale,
                "restoreformer_restoration_level": self.face_enhancement.restoreformer_restoration_level,
                "wan_model_type": self.face_enhancement.wan_model_type
            },
            "inpainting": {
                "primary_method": self.inpainting.primary_method.value,
                "fallback_method": self.inpainting.fallback_method.value,
                "enable_smart_detection": self.inpainting.enable_smart_detection,
                "enable_glasses_detection": self.inpainting.enable_glasses_detection,
                "enable_hair_detection": self.inpainting.enable_hair_detection,
                "enable_mask_detection": self.inpainting.enable_mask_detection,
                "boundary_expand_ratio": self.inpainting.boundary_expand_ratio,
                "mask_feather_radius": self.inpainting.mask_feather_radius,
                "inpaint_radius": self.inpainting.inpaint_radius,
                "enable_edge_preservation": self.inpainting.enable_edge_preservation,
                "context_search_radius": self.inpainting.context_search_radius,
                "enable_multi_pass": self.inpainting.enable_multi_pass,
                "max_iterations": self.inpainting.max_iterations
            },
            "face_profile": {
                "enable_quality_analysis": self.face_profile.enable_quality_analysis,
                "quality_weights": self.face_profile.quality_weights,
                "minimum_quality_score": self.face_profile.minimum_quality_score,
                "prefer_frontal_faces": self.face_profile.prefer_frontal_faces,
                "max_pose_angle": self.face_profile.max_pose_angle,
                "enable_multi_angle_profiling": self.face_profile.enable_multi_angle_profiling,
                "generate_face_variations": self.face_profile.generate_face_variations,
                "max_variations": self.face_profile.max_variations,
                "enable_pose_estimation": self.face_profile.enable_pose_estimation,
                "pose_estimation_method": self.face_profile.pose_estimation_method
            },
            "performance": {
                "priority": self.performance.priority.value,
                "enable_memory_optimization": self.performance.enable_memory_optimization,
                "max_memory_usage_gb": self.performance.max_memory_usage_gb,
                "enable_garbage_collection": self.performance.enable_garbage_collection,
                "enable_gpu_acceleration": self.performance.enable_gpu_acceleration,
                "gpu_memory_fraction": self.performance.gpu_memory_fraction,
                "enable_mixed_precision": self.performance.enable_mixed_precision,
                "max_worker_threads": self.performance.max_worker_threads,
                "enable_parallel_processing": self.performance.enable_parallel_processing,
                "batch_size": self.performance.batch_size,
                "enable_batch_optimization": self.performance.enable_batch_optimization
            },
            "global": {
                "enable_enhanced_features": self.enable_enhanced_features,
                "enable_logging": self.enable_logging,
                "log_level": self.log_level,
                "maintain_backwards_compatibility": self.maintain_backwards_compatibility,
                "enable_fallback_methods": self.enable_fallback_methods
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedProcessingConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update video config
        if "video" in config_dict:
            video_config = config_dict["video"]
            config.video.enable_frame_interpolation = video_config.get("enable_frame_interpolation", False)
            config.video.target_fps = video_config.get("target_fps")
            if "interpolation_method" in video_config:
                config.video.interpolation_method = InterpolationMethod(video_config["interpolation_method"])
            config.video.max_interpolation_factor = video_config.get("max_interpolation_factor", 4)
            config.video.enable_temporal_consistency = video_config.get("enable_temporal_consistency", True)
            config.video.temporal_smoothing_factor = video_config.get("temporal_smoothing_factor", 0.3)
            config.video.enable_optical_flow = video_config.get("enable_optical_flow", True)
            config.video.flow_quality = video_config.get("flow_quality", "medium")
            config.video.maintain_aspect_ratio = video_config.get("maintain_aspect_ratio", True)
            config.video.enable_quality_optimization = video_config.get("enable_quality_optimization", True)
            config.video.adaptive_quality = video_config.get("adaptive_quality", True)
        
        # Update face enhancement config
        if "face_enhancement" in config_dict:
            fe_config = config_dict["face_enhancement"]
            if "primary_model" in fe_config:
                config.face_enhancement.primary_model = FaceEnhancementModel(fe_config["primary_model"])
            if "fallback_model" in fe_config:
                config.face_enhancement.fallback_model = FaceEnhancementModel(fe_config["fallback_model"])
            config.face_enhancement.enhancement_level = fe_config.get("enhancement_level", 0.8)
            config.face_enhancement.upscale_factor = fe_config.get("upscale_factor", 2)
            config.face_enhancement.minimum_face_quality = fe_config.get("minimum_face_quality", 0.4)
            config.face_enhancement.target_face_quality = fe_config.get("target_face_quality", 0.8)
            config.face_enhancement.enable_preprocessing = fe_config.get("enable_preprocessing", True)
            config.face_enhancement.enable_postprocessing = fe_config.get("enable_postprocessing", True)
            config.face_enhancement.preserve_original_lighting = fe_config.get("preserve_original_lighting", True)
            config.face_enhancement.realesrgan_tile_size = fe_config.get("realesrgan_tile_size", 512)
            config.face_enhancement.realesrgan_scale = fe_config.get("realesrgan_scale", 4)
            config.face_enhancement.restoreformer_restoration_level = fe_config.get("restoreformer_restoration_level", 0.8)
            config.face_enhancement.wan_model_type = fe_config.get("wan_model_type", "enhancement")
        
        # Update inpainting config
        if "inpainting" in config_dict:
            inp_config = config_dict["inpainting"]
            if "primary_method" in inp_config:
                config.inpainting.primary_method = InpaintingMethod(inp_config["primary_method"])
            if "fallback_method" in inp_config:
                config.inpainting.fallback_method = InpaintingMethod(inp_config["fallback_method"])
            config.inpainting.enable_smart_detection = inp_config.get("enable_smart_detection", True)
            config.inpainting.enable_glasses_detection = inp_config.get("enable_glasses_detection", True)
            config.inpainting.enable_hair_detection = inp_config.get("enable_hair_detection", True)
            config.inpainting.enable_mask_detection = inp_config.get("enable_mask_detection", True)
            config.inpainting.boundary_expand_ratio = inp_config.get("boundary_expand_ratio", 0.1)
            config.inpainting.mask_feather_radius = inp_config.get("mask_feather_radius", 5)
            config.inpainting.inpaint_radius = inp_config.get("inpaint_radius", 3)
            config.inpainting.enable_edge_preservation = inp_config.get("enable_edge_preservation", True)
            config.inpainting.context_search_radius = inp_config.get("context_search_radius", 20)
            config.inpainting.enable_multi_pass = inp_config.get("enable_multi_pass", False)
            config.inpainting.max_iterations = inp_config.get("max_iterations", 3)
        
        # Update face profile config
        if "face_profile" in config_dict:
            fp_config = config_dict["face_profile"]
            config.face_profile.enable_quality_analysis = fp_config.get("enable_quality_analysis", True)
            if "quality_weights" in fp_config:
                config.face_profile.quality_weights.update(fp_config["quality_weights"])
            config.face_profile.minimum_quality_score = fp_config.get("minimum_quality_score", 0.6)
            config.face_profile.prefer_frontal_faces = fp_config.get("prefer_frontal_faces", True)
            config.face_profile.max_pose_angle = fp_config.get("max_pose_angle", 30.0)
            config.face_profile.enable_multi_angle_profiling = fp_config.get("enable_multi_angle_profiling", True)
            config.face_profile.generate_face_variations = fp_config.get("generate_face_variations", True)
            config.face_profile.max_variations = fp_config.get("max_variations", 5)
            config.face_profile.enable_pose_estimation = fp_config.get("enable_pose_estimation", True)
            config.face_profile.pose_estimation_method = fp_config.get("pose_estimation_method", "pnp")
        
        # Update performance config
        if "performance" in config_dict:
            perf_config = config_dict["performance"]
            if "priority" in perf_config:
                config.performance.priority = ProcessingPriority(perf_config["priority"])
            config.performance.enable_memory_optimization = perf_config.get("enable_memory_optimization", True)
            config.performance.max_memory_usage_gb = perf_config.get("max_memory_usage_gb", 8.0)
            config.performance.enable_garbage_collection = perf_config.get("enable_garbage_collection", True)
            config.performance.enable_gpu_acceleration = perf_config.get("enable_gpu_acceleration", True)
            config.performance.gpu_memory_fraction = perf_config.get("gpu_memory_fraction", 0.8)
            config.performance.enable_mixed_precision = perf_config.get("enable_mixed_precision", False)
            config.performance.max_worker_threads = perf_config.get("max_worker_threads", 4)
            config.performance.enable_parallel_processing = perf_config.get("enable_parallel_processing", True)
            config.performance.batch_size = perf_config.get("batch_size", 1)
            config.performance.enable_batch_optimization = perf_config.get("enable_batch_optimization", False)
        
        # Update global config
        if "global" in config_dict:
            global_config = config_dict["global"]
            config.enable_enhanced_features = global_config.get("enable_enhanced_features", True)
            config.enable_logging = global_config.get("enable_logging", True)
            config.log_level = global_config.get("log_level", "INFO")
            config.maintain_backwards_compatibility = global_config.get("maintain_backwards_compatibility", True)
            config.enable_fallback_methods = global_config.get("enable_fallback_methods", True)
        
        return config
    
    def get_preset_config(self, preset: str) -> 'EnhancedProcessingConfig':
        """Get predefined configuration presets."""
        if preset == "speed":
            return self._get_speed_preset()
        elif preset == "quality":
            return self._get_quality_preset()
        elif preset == "balanced":
            return self._get_balanced_preset()
        else:
            logger.warning(f"Unknown preset: {preset}, using balanced")
            return self._get_balanced_preset()
    
    def _get_speed_preset(self) -> 'EnhancedProcessingConfig':
        """Speed-optimized configuration."""
        config = EnhancedProcessingConfig()
        config.performance.priority = ProcessingPriority.SPEED
        config.video.enable_frame_interpolation = False
        config.video.interpolation_method = InterpolationMethod.SIMPLE
        config.face_enhancement.primary_model = FaceEnhancementModel.GFPGAN
        config.face_enhancement.enhancement_level = 0.5
        config.inpainting.primary_method = InpaintingMethod.TRADITIONAL_TELEA
        config.inpainting.enable_smart_detection = False
        config.face_profile.enable_quality_analysis = False
        return config
    
    def _get_quality_preset(self) -> 'EnhancedProcessingConfig':
        """Quality-optimized configuration."""
        config = EnhancedProcessingConfig()
        config.performance.priority = ProcessingPriority.QUALITY
        config.video.enable_frame_interpolation = True
        config.video.interpolation_method = InterpolationMethod.RIFE
        config.face_enhancement.primary_model = FaceEnhancementModel.REAL_ESRGAN
        config.face_enhancement.enhancement_level = 1.0
        config.face_enhancement.upscale_factor = 4
        config.inpainting.primary_method = InpaintingMethod.CONTEXT_AWARE
        config.inpainting.enable_multi_pass = True
        config.face_profile.enable_quality_analysis = True
        config.face_profile.enable_multi_angle_profiling = True
        return config
    
    def _get_balanced_preset(self) -> 'EnhancedProcessingConfig':
        """Balanced configuration (default)."""
        return EnhancedProcessingConfig()


# Global configuration instance
_global_config = None


def get_enhanced_config() -> EnhancedProcessingConfig:
    """Get the global enhanced processing configuration."""
    global _global_config
    if _global_config is None:
        _global_config = EnhancedProcessingConfig()
    return _global_config


def set_enhanced_config(config: EnhancedProcessingConfig):
    """Set the global enhanced processing configuration."""
    global _global_config
    _global_config = config


def load_config_from_file(file_path: str) -> EnhancedProcessingConfig:
    """Load configuration from file."""
    try:
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return EnhancedProcessingConfig.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {e}")
        return EnhancedProcessingConfig()


def save_config_to_file(config: EnhancedProcessingConfig, file_path: str):
    """Save configuration to file."""
    try:
        import json
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {file_path}: {e}")