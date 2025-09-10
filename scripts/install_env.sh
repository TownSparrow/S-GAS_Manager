#!/bin/bash

### Script for installing the environment and all needed dependencies###

echo "1. Creating Python virtual environment..."
python3 -m venv S-GAS_Manager_env

echo "2. Activating the environment..."
source S-GAS_Manager_env/bin/activate

echo "3. Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

echo "4. Installing core dependencies from requirements.txt..."
pip install -r other/requirements.txt

echo "5. Downloading spaCy model for Russian (medium model, more accurate)..."
python -m spacy download ru_core_news_md

echo "Installation complete! Don't forget to activate the environment with: source S-GAS_Manager_env/bin/activate"