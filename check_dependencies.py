#!/usr/bin/env python3
import subprocess
import sys
import importlib

def check_and_install_dependency(package_name, install_name=None):
    """Check if a package is installed, and install it if not."""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False

def main():
    """Check and install all dependencies required for process_pdfs.py."""
    print("Checking and installing dependencies...")
    
    # Define the dependencies needed
    dependencies = [
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("docx", "python-docx"),
        ("pdfplumber", "pdfplumber"),
        ("fuzzywuzzy", "fuzzywuzzy[speedup]"),  # installs python-Levenshtein for speed
    ]
    
    all_installed = True
    for package_name, install_name in dependencies:
        success = check_and_install_dependency(package_name, install_name)
        all_installed = all_installed and success
    
    if all_installed:
        print("\nAll dependencies are installed! You can now run process_pdfs.py")
        return 0
    else:
        print("\nSome dependencies could not be installed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 