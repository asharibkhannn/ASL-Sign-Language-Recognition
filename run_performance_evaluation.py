#!/usr/bin/env python3
"""
ASL Sign Language Recognition - Performance Evaluation Launcher
This script provides an easy way to run the performance evaluation on any platform.
"""

import os
import sys
import subprocess
import argparse
import platform

def check_dependencies():
    """Check if required dependencies are installed and install them if needed"""
    try:
        import tensorflow
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
        import pandas
        import cv2
        import PIL
        print("All required packages are already installed.")
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "tensorflow", "scikit-learn", "matplotlib", "seaborn", 
            "tqdm", "pandas", "opencv-python", "pillow"
        ])
        print("Required packages installed successfully.")

def check_model_files():
    """Check if the model and class mapping files exist"""
    model_found = False
    for model_name in ['asl_classifier_finetuned.h5', 'asl_classifier_best.h5', 'asl_classifier_final.h5']:
        if os.path.exists(model_name):
            model_found = True
            print(f"Found model file: {model_name}")
            break
    
    if not model_found:
        print("Error: No model file found!")
        print("Please make sure one of the following files is in the current directory:")
        print("- asl_classifier_finetuned.h5")
        print("- asl_classifier_best.h5")
        print("- asl_classifier_final.h5")
        return False
    
    if not os.path.exists('class_mapping.npy'):
        print("Error: Class mapping file (class_mapping.npy) not found!")
        print("Please make sure the class mapping file is in the current directory.")
        return False
    
    return True

def get_input(prompt):
    """Get user input with the given prompt"""
    try:
        return input(prompt)
    except EOFError:
        return ''

def run_evaluation():
    """Run the performance evaluation script"""
    # Check if required files exist
    if not check_model_files():
        return False
    
    # Install dependencies if needed
    check_dependencies()
    
    # Parse command line arguments if provided
    parser = argparse.ArgumentParser(description='Run ASL Sign Language Recognition performance evaluation')
    parser.add_argument('--test_dir', type=str, help='Directory containing test images organized by class')
    parser.add_argument('--test_image', type=str, help='Single test image for preprocessing comparison')
    args = parser.parse_args()
    
    # Get inputs from user if not provided as arguments
    test_dir = args.test_dir
    test_image = args.test_image
    
    if test_dir is None and test_image is None:
        print("\nASL Sign Language Recognition - Performance Evaluation")
        print("="*56)
        
        test_dir = get_input("\nEnter the path to test data directory (leave empty to skip): ")
        test_image = get_input("Enter the path to a test image for preprocessing comparison (leave empty to skip): ")
    
    # Build the command
    cmd = [sys.executable, 'evaluate_model_performance.py']
    if test_dir:
        cmd.extend(['--test_dir', test_dir])
    if test_image:
        cmd.extend(['--test_image', test_image])
    
    # If no inputs are provided, show help
    if not test_dir and not test_image:
        print("\nNo inputs provided. At least one of the following is required:")
        print("- Test data directory (containing subdirectories for each class)")
        print("- Test image file (for preprocessing comparison)")
        print("\nUsage examples:")
        print("- Evaluate on test data: Provide a directory containing subdirectories for each ASL letter")
        print("- Compare preprocessing methods: Provide a single test image")
        print("\nThe script will generate graphs and a detailed performance report.")
        return False
    
    # Run the evaluation script
    print("\nRunning performance evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nEvaluation completed successfully.")
        
        # Show results location
        if os.path.exists('performance_results'):
            print("Performance results are available in the 'performance_results' folder.")
            print("Open performance_results/performance_report.md for a summary.")
        if os.path.exists('preprocessing_comparison.png'):
            print("Preprocessing comparison image saved as preprocessing_comparison.png")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during evaluation: {e}")
        return False

if __name__ == "__main__":
    success = run_evaluation()
    
    # On Windows, keep console window open
    if platform.system() == "Windows" and not sys.stdout.isatty():
        input("\nPress Enter to exit...")
    
    sys.exit(0 if success else 1) 