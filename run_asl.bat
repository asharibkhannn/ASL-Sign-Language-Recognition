@echo off
echo Launching ASL Sign Detector...
python3 asl_camera_gui.py
if errorlevel 1 (
    echo Trying with 'python' command...
    python asl_camera_gui.py
)
pause 