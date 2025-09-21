#!/bin/bash

# ========================================
# SYSTEM UPDATE & DEPENDENCIES
# ========================================
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget git git-lfs python3 python3-pip unzip jq htop cmake \
    openjdk-11-jdk repo ccache zsh fzf ripgrep
git lfs install

# ccache for Android builds
echo 'export USE_CCACHE=1' >> ~/.bashrc
echo 'ccache -M 50G' >> ~/.bashrc

# ========================================
# OLLAMA MODEL DIRECTORY
# ========================================
sudo mkdir -p /mnt/d/ollama_models
sudo chown "$USER":"$USER" /mnt/d/ollama_models
export OLLAMA_DIR=/mnt/d/ollama_models
echo "export OLLAMA_DIR=/mnt/d/ollama_models" >> ~/.bashrc

# ========================================
# INSTALL OLLAMA
# ========================================
curl -fsSL https://ollama.com/install.sh | sh
ollama --version

# ========================================
# PULL MODELS
# ========================================
ollama pull gemma3:27b-it-qat
ollama pull granite-code:8b-instruct
ollama pull mistral:7b
ollama pull llama3.2-vision:11b
ollama pull deepseek-coder:6.7b-instruct-q8_0
ollama run phi3:14b-medium-128k-instruct-q4_0

# ========================================
# SETUP ANOMZ
# ========================================
mkdir -p ~/anomz
cd ~/anomz
pip install -r requirements.txt

echo "Setup complete! Launch Anomz using:"
echo "python app.py"
