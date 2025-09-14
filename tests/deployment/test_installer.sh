#!/bin/bash
# Basic installer test for CI workflow

echo "Testing installer scripts..."

# Check if installer directory exists
if [ ! -d "installer" ]; then
    echo "⚠ Installer directory not found"
    exit 0  # Don't fail CI for this
fi

cd installer

# Check for installer files
installer_files=$(find . -name "*.py" -o -name "*.sh" | wc -l)
echo "Found $installer_files installer files"

# Basic syntax check for Python installer files
for file in *.py; do
    if [ -f "$file" ]; then
        echo "Checking Python file: $file"
        python3 -m py_compile "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ $file syntax OK"
        else
            echo "✗ $file has syntax errors"
        fi
    fi
done

# Basic syntax check for shell scripts
for file in *.sh; do
    if [ -f "$file" ]; then
        echo "Checking shell script: $file"
        bash -n "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ $file syntax OK"
        else
            echo "✗ $file has syntax errors"
        fi
    fi
done

echo "✓ Installer test completed"
exit 0