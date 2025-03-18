# ASL Sign Detector

A real-time American Sign Language (ASL) detection application that recognizes hand signs using a webcam.

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

## Files

- `asl_camera_gui.py` - Main application file
- `run_asl.py` - Python launcher script
- `run_asl.bat` - Windows batch launcher
- `asl_classifier_finetuned.h5` - Trained ASL classification model
- `class_mapping.npy` - Mapping of class indices to ASL letters
- `requirements.txt` - Required dependencies 