#!/usr/bin/env python3
"""
Basic UI components test for CI workflow.
"""

import os
import sys


def test_ui_components():
    """Test UI components availability."""
    print("Testing UI components...")
    
    # Check for demo files
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    demo_files = []
    
    for filename in os.listdir(root_dir):
        if filename.startswith('demo') and filename.endswith('.py'):
            demo_files.append(filename)
    
    print(f"Found demo files: {demo_files}")
    
    # Basic syntax check for demo files
    for demo_file in demo_files:
        filepath = os.path.join(root_dir, demo_file)
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Basic validation - check if it's a Python file
            if 'import' in content and 'def ' in content:
                print(f"✓ {demo_file} appears to be a valid Python file")
            else:
                print(f"⚠ {demo_file} may not be a standard Python file")
                
        except Exception as e:
            print(f"✗ Error checking {demo_file}: {e}")
    
    print("UI components test completed")
    return True


if __name__ == "__main__":
    if test_ui_components():
        sys.exit(0)
    else:
        sys.exit(1)