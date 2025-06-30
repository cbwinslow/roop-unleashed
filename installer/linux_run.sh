#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$SCRIPT_DIR/installer_files"
CONDA_ROOT_PREFIX="$INSTALL_DIR/conda"
INSTALL_ENV_DIR="$INSTALL_DIR/env"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

if [ ! -x "$CONDA_ROOT_PREFIX/bin/conda" ]; then
    echo "Downloading Miniconda..."
    mkdir -p "$INSTALL_DIR"
    curl -Lk "$MINICONDA_URL" -o "$INSTALL_DIR/miniconda.sh"
    bash "$INSTALL_DIR/miniconda.sh" -b -p "$CONDA_ROOT_PREFIX"
fi

if [ ! -d "$INSTALL_ENV_DIR" ]; then
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" python=3.10
fi

source "$CONDA_ROOT_PREFIX/bin/activate" "$INSTALL_ENV_DIR"

# install ffmpeg if missing
if ! command -v ffmpeg >/dev/null 2>&1; then
    "$CONDA_ROOT_PREFIX/bin/conda" install -y -c conda-forge ffmpeg
fi

# install torch with cuda support and other requirements
"$CONDA_ROOT_PREFIX/bin/conda" install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8
python -m pip install -r "$REPO_DIR/requirements.txt"

cd "$REPO_DIR"
python run.py "$@"
