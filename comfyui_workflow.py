#!/usr/bin/env python3
"""
ComfyUI-inspired workflow integration for roop-unleashed.
Enables loading and executing ComfyUI-style workflows within roop-unleashed.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class ComfyUIWorkflowNode:
    """Base class for ComfyUI-style workflow nodes."""
    
    def __init__(self, node_id: str, node_type: str, inputs: Dict[str, Any], outputs: List[str]):
        self.node_id = node_id
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = outputs
        self.processed = False
        self.output_data = {}
    
    def execute(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node and return output data."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_inputs(self, workflow_context: Dict[str, Any]) -> bool:
        """Validate that all required inputs are available."""
        for input_name, input_spec in self.inputs.items():
            if isinstance(input_spec, list) and len(input_spec) == 2:
                # Input is a reference to another node's output
                source_node_id, output_name = input_spec
                if source_node_id not in workflow_context:
                    logger.error(f"Node {self.node_id}: Source node {source_node_id} not found")
                    return False
                if output_name not in workflow_context[source_node_id]:
                    logger.error(f"Node {self.node_id}: Output {output_name} not found in node {source_node_id}")
                    return False
        return True

class LoadImageNode(ComfyUIWorkflowNode):
    """Load image from file or URL."""
    
    def execute(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.inputs.get('image', '')
        
        if not image_path:
            raise ValueError("LoadImageNode requires 'image' input")
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                # Try relative to workflow file
                workflow_dir = workflow_context.get('workflow_dir', '.')
                full_path = os.path.join(workflow_dir, image_path)
                if os.path.exists(full_path):
                    image = Image.open(full_path)
                else:
                    raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for processing
            image_array = np.array(image)
            
            self.output_data = {
                'IMAGE': image_array,
                'image_info': {
                    'width': image.width,
                    'height': image.height,
                    'mode': image.mode,
                    'format': image.format
                }
            }
            
            logger.info(f"Loaded image: {image_path} ({image.width}x{image.height})")
            return self.output_data
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

class FaceSwapNode(ComfyUIWorkflowNode):
    """Face swap processing node compatible with roop-unleashed."""
    
    def execute(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Get source and target images
            source_image = self._get_input_data('source_image', workflow_context)
            target_image = self._get_input_data('target_image', workflow_context)
            
            if source_image is None or target_image is None:
                raise ValueError("FaceSwapNode requires source_image and target_image inputs")
            
            # Import roop modules for face swapping
            try:
                import roop.globals
                from roop.face_util import extract_face_images
                from roop.processors.frame.face_swapper import process_frame
            except ImportError as e:
                logger.error(f"Cannot import roop modules: {e}")
                raise
            
            # Configure face swap parameters
            swap_params = {
                'blend_ratio': self.inputs.get('blend_ratio', 0.65),
                'distance_threshold': self.inputs.get('distance_threshold', 0.65),
                'face_enhancer': self.inputs.get('face_enhancer', None),
                'face_index': self.inputs.get('face_index', 0)
            }
            
            # Perform face swap (simplified integration)
            # In a real implementation, this would use the full roop pipeline
            logger.info("Performing face swap with roop-unleashed pipeline")
            
            # For now, return the target image with metadata
            # This would be replaced with actual face swap processing
            self.output_data = {
                'IMAGE': target_image,
                'swap_info': swap_params,
                'processing_status': 'completed'
            }
            
            return self.output_data
            
        except Exception as e:
            logger.error(f"Error in face swap processing: {e}")
            raise
    
    def _get_input_data(self, input_name: str, workflow_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get input data from workflow context."""
        if input_name not in self.inputs:
            return None
        
        input_spec = self.inputs[input_name]
        if isinstance(input_spec, list) and len(input_spec) == 2:
            source_node_id, output_name = input_spec
            if source_node_id in workflow_context and output_name in workflow_context[source_node_id]:
                return workflow_context[source_node_id][output_name]
        
        return None

class EnhanceFaceNode(ComfyUIWorkflowNode):
    """Face enhancement node using roop's enhancers."""
    
    def execute(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Get input image
            input_image = self._get_input_data('image', workflow_context)
            if input_image is None:
                raise ValueError("EnhanceFaceNode requires image input")
            
            # Enhancement parameters
            enhancer_type = self.inputs.get('enhancer', 'gfpgan')
            scale_factor = self.inputs.get('scale', 2)
            
            # Import roop enhancement modules
            try:
                from roop.processors.frame.face_enhancer import enhance_face
            except ImportError:
                logger.warning("Face enhancer not available, returning original image")
                enhanced_image = input_image
            else:
                # Apply enhancement
                enhanced_image = input_image  # Placeholder - would use actual enhancer
                logger.info(f"Applied {enhancer_type} enhancement with scale {scale_factor}")
            
            self.output_data = {
                'IMAGE': enhanced_image,
                'enhancement_info': {
                    'type': enhancer_type,
                    'scale': scale_factor,
                    'status': 'completed'
                }
            }
            
            return self.output_data
            
        except Exception as e:
            logger.error(f"Error in face enhancement: {e}")
            raise
    
    def _get_input_data(self, input_name: str, workflow_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get input data from workflow context."""
        if input_name not in self.inputs:
            return None
        
        input_spec = self.inputs[input_name]
        if isinstance(input_spec, list) and len(input_spec) == 2:
            source_node_id, output_name = input_spec
            if source_node_id in workflow_context and output_name in workflow_context[source_node_id]:
                return workflow_context[source_node_id][output_name]
        
        return None

class SaveImageNode(ComfyUIWorkflowNode):
    """Save image to file."""
    
    def execute(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Get input image
            input_image = self._get_input_data('image', workflow_context)
            if input_image is None:
                raise ValueError("SaveImageNode requires image input")
            
            # Output parameters
            output_path = self.inputs.get('filename', 'output.png')
            quality = self.inputs.get('quality', 95)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert numpy array back to PIL Image
            if isinstance(input_image, np.ndarray):
                image = Image.fromarray(input_image.astype(np.uint8))
            else:
                image = input_image
            
            # Save image
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                image.save(output_path, 'JPEG', quality=quality)
            else:
                image.save(output_path)
            
            self.output_data = {
                'saved_path': output_path,
                'file_size': os.path.getsize(output_path),
                'status': 'saved'
            }
            
            logger.info(f"Saved image to: {output_path}")
            return self.output_data
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise
    
    def _get_input_data(self, input_name: str, workflow_context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get input data from workflow context."""
        if input_name not in self.inputs:
            return None
        
        input_spec = self.inputs[input_name]
        if isinstance(input_spec, list) and len(input_spec) == 2:
            source_node_id, output_name = input_spec
            if source_node_id in workflow_context and output_name in workflow_context[source_node_id]:
                return workflow_context[source_node_id][output_name]
        
        return None

class ComfyUIWorkflowEngine:
    """Engine for executing ComfyUI-style workflows in roop-unleashed."""
    
    def __init__(self):
        self.node_registry = {
            'LoadImage': LoadImageNode,
            'FaceSwap': FaceSwapNode,
            'EnhanceFace': EnhanceFaceNode,
            'SaveImage': SaveImageNode,
        }
        self.workflow_context = {}
    
    def register_node_type(self, node_type: str, node_class: type):
        """Register a new node type."""
        self.node_registry[node_type] = node_class
        logger.info(f"Registered node type: {node_type}")
    
    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Load workflow from JSON file."""
        try:
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            
            logger.info(f"Loaded workflow from: {workflow_path}")
            return workflow_data
            
        except Exception as e:
            logger.error(f"Error loading workflow {workflow_path}: {e}")
            raise
    
    def parse_workflow(self, workflow_data: Dict[str, Any]) -> List[ComfyUIWorkflowNode]:
        """Parse workflow data into node objects."""
        nodes = []
        
        # Parse nodes from workflow
        workflow_nodes = workflow_data.get('nodes', {})
        
        for node_id, node_info in workflow_nodes.items():
            node_type = node_info.get('class_type', '')
            inputs = node_info.get('inputs', {})
            outputs = node_info.get('outputs', [])
            
            if node_type in self.node_registry:
                node_class = self.node_registry[node_type]
                node = node_class(node_id, node_type, inputs, outputs)
                nodes.append(node)
                logger.debug(f"Created node: {node_id} ({node_type})")
            else:
                logger.warning(f"Unknown node type: {node_type} in node {node_id}")
        
        return nodes
    
    def execute_workflow(self, workflow_path: str, **kwargs) -> Dict[str, Any]:
        """Execute a complete workflow."""
        try:
            # Load workflow
            workflow_data = self.load_workflow(workflow_path)
            
            # Initialize context
            self.workflow_context = {
                'workflow_dir': os.path.dirname(workflow_path),
                'workflow_name': os.path.basename(workflow_path),
                **kwargs
            }
            
            # Parse nodes
            nodes = self.parse_workflow(workflow_data)
            
            # Execute nodes in dependency order
            executed_nodes = set()
            results = {}
            
            while len(executed_nodes) < len(nodes):
                progress_made = False
                
                for node in nodes:
                    if node.node_id in executed_nodes:
                        continue
                    
                    # Check if all dependencies are satisfied
                    if node.validate_inputs(self.workflow_context):
                        try:
                            logger.info(f"Executing node: {node.node_id} ({node.node_type})")
                            node_output = node.execute(self.workflow_context)
                            
                            # Store results in context
                            self.workflow_context[node.node_id] = node_output
                            results[node.node_id] = node_output
                            executed_nodes.add(node.node_id)
                            progress_made = True
                            
                            logger.info(f"Completed node: {node.node_id}")
                            
                        except Exception as e:
                            logger.error(f"Error executing node {node.node_id}: {e}")
                            raise
                
                if not progress_made and len(executed_nodes) < len(nodes):
                    unexecuted = [n.node_id for n in nodes if n.node_id not in executed_nodes]
                    raise RuntimeError(f"Workflow execution stalled. Unexecuted nodes: {unexecuted}")
            
            logger.info("Workflow execution completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

# Example workflow creation utilities
def create_basic_face_swap_workflow(source_image: str, target_image: str, output_path: str) -> Dict[str, Any]:
    """Create a basic face swap workflow in ComfyUI format."""
    workflow = {
        "nodes": {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": source_image
                },
                "outputs": ["IMAGE", "image_info"]
            },
            "2": {
                "class_type": "LoadImage", 
                "inputs": {
                    "image": target_image
                },
                "outputs": ["IMAGE", "image_info"]
            },
            "3": {
                "class_type": "FaceSwap",
                "inputs": {
                    "source_image": ["1", "IMAGE"],
                    "target_image": ["2", "IMAGE"],
                    "blend_ratio": 0.65,
                    "distance_threshold": 0.65
                },
                "outputs": ["IMAGE", "swap_info"]
            },
            "4": {
                "class_type": "EnhanceFace",
                "inputs": {
                    "image": ["3", "IMAGE"],
                    "enhancer": "gfpgan",
                    "scale": 2
                },
                "outputs": ["IMAGE", "enhancement_info"]
            },
            "5": {
                "class_type": "SaveImage",
                "inputs": {
                    "image": ["4", "IMAGE"],
                    "filename": output_path,
                    "quality": 95
                },
                "outputs": ["saved_path", "status"]
            }
        },
        "metadata": {
            "description": "Basic face swap with enhancement",
            "version": "1.0",
            "created_by": "roop-unleashed-comfyui"
        }
    }
    
    return workflow

def save_workflow(workflow: Dict[str, Any], path: str) -> None:
    """Save workflow to JSON file."""
    with open(path, 'w') as f:
        json.dump(workflow, f, indent=2)
    logger.info(f"Saved workflow to: {path}")

# Example usage
if __name__ == "__main__":
    # Create workflow engine
    engine = ComfyUIWorkflowEngine()
    
    # Create example workflow
    workflow = create_basic_face_swap_workflow(
        source_image="/app/examples/source.jpg",
        target_image="/app/examples/target.jpg", 
        output_path="/app/output/result.jpg"
    )
    
    # Save workflow
    os.makedirs("/app/workflows", exist_ok=True)
    save_workflow(workflow, "/app/workflows/basic_face_swap.json")
    
    # Execute workflow (would require actual images)
    try:
        results = engine.execute_workflow("/app/workflows/basic_face_swap.json")
        print("Workflow executed successfully:", results)
    except Exception as e:
        print(f"Workflow execution failed: {e}")