# ASL Sign Language Recognition

A real-time American Sign Language (ASL) detection application that recognizes hand signs using a webcam. This project utilizes Machine Learning and deep learning LSTM model for accurate sign language recognition.

## Features

- Real-time ASL hand sign detection (A-Z)
- User-friendly graphical interface
- Adjustable confidence threshold
- Visual feedback with confidence meter
- Enhanced hand detection using skin segmentation

## Requirements

```
tensorflow>=2.10.0
opencv-python>=4.6.0
numpy>=1.21.0
pillow>=9.0.0
```

## How to Run

### Easy Method (Recommended)

**Windows Users:**
- Double-click on `run_asl.bat`
- Or run in a terminal: `.\run_asl.bat`

**Linux/Mac Users:**
- Make the Python launcher executable: `chmod +x run_asl.py`
- Run: `./run_asl.py`

### Manual Method

1. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python3 asl_camera_gui.py
   ```

## Usage Instructions

1. Position your hand in the green rectangle in the center of the frame
2. Make an ASL hand sign
3. Hold the sign steady for best results
4. Use the confidence threshold slider to adjust sensitivity
5. Close the window to exit the application

## Performance Evaluation

This project includes tools to measure and visualize the performance of the ASL Sign Detection model:

### Running Performance Evaluation

**Windows Users:**
- Double-click on `run_performance_evaluation.bat`
- Or run in a terminal: `.\run_performance_evaluation.bat`

**Linux/Mac Users:**
- Make the script executable: `chmod +x run_performance_evaluation.py`
- Run: `./run_performance_evaluation.py`

### What It Measures

The performance evaluation generates the following metrics and visualizations:

1. **Overall Metrics**:
   - Accuracy, Precision, Recall, and F1 Score
   - Average confidence scores
   - Inference times
   
2. **Per-Class Analysis**:
   - Individual accuracy for each ASL letter
   - Confusion matrix showing misclassifications
   - Identification of best and worst-performing signs
   
3. **Preprocessing Effectiveness**:
   - Comparison of different image preprocessing techniques
   - Analysis of skin segmentation methods
   
4. **Confidence Threshold Analysis**:
   - Effect of different confidence thresholds on accuracy
   - Optimal threshold determination

### Using the Results

The evaluation produces a comprehensive report and visualizations in the `performance_results` folder, which can help:
- Identify ASL signs that need improvement
- Optimize preprocessing for better hand detection
- Fine-tune the confidence threshold for optimal accuracy

## Files

- `asl_camera_gui.py` - Main application file
- `run_asl.py` - Python launcher script
- `run_asl.bat` - Windows batch launcher
- `asl_classifier_finetuned.h5` - Trained ASL classification model
- `class_mapping.npy` - Mapping of class indices to ASL letters
- `requirements.txt` - Required dependencies
- `evaluate_model_performance.py` - Performance evaluation script
- `run_performance_evaluation.py` - Python launcher for evaluation
- `run_performance_evaluation.bat` - Windows batch launcher for evaluation
