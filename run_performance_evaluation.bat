@echo off
echo ASL Sign Language Recognition - Performance Evaluation
echo =================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.x and try again.
    goto :error
)

REM Check if the model file exists
if not exist asl_classifier_finetuned.h5 (
    echo Model file (asl_classifier_finetuned.h5) not found!
    echo Please make sure the model file is in the current directory.
    goto :error
)

REM Check if the class mapping file exists
if not exist class_mapping.npy (
    echo Class mapping file (class_mapping.npy) not found!
    echo Please make sure the class mapping file is in the current directory.
    goto :error
)

REM Install required packages if not already installed
echo Installing required packages...
pip install -q tensorflow scikit-learn matplotlib seaborn tqdm pandas opencv-python pillow

REM Check if test directory is provided
set TEST_DIR=
set /p TEST_DIR="Enter the path to test data directory (leave empty to skip): "

REM Check if test image is provided
set TEST_IMAGE=
set /p TEST_IMAGE="Enter the path to a test image for preprocessing comparison (leave empty to skip): "

REM Build the command based on inputs
set CMD=python evaluate_model_performance.py
if not "%TEST_DIR%"=="" (
    set CMD=%CMD% --test_dir "%TEST_DIR%"
)
if not "%TEST_IMAGE%"=="" (
    set CMD=%CMD% --test_image "%TEST_IMAGE%"
)

REM If no inputs are provided, show help
if "%TEST_DIR%%TEST_IMAGE%"=="" (
    echo.
    echo No inputs provided. At least one of the following is required:
    echo - Test data directory (containing subdirectories for each class)
    echo - Test image file (for preprocessing comparison)
    goto :help
)

REM Run the evaluation script
echo.
echo Running performance evaluation...
echo Command: %CMD%
echo.
%CMD%

REM Show results location
echo.
if exist performance_results (
    echo Performance results are available in the "performance_results" folder.
    echo Open performance_results\performance_report.md for a summary.
)
if exist preprocessing_comparison.png (
    echo Preprocessing comparison image saved as preprocessing_comparison.png
)

goto :end

:help
echo.
echo Usage examples:
echo - Evaluate on test data: Provide a directory containing subdirectories for each ASL letter
echo - Compare preprocessing methods: Provide a single test image
echo.
echo The script will generate graphs and a detailed performance report.
goto :end

:error
echo.
echo Error occurred during execution.
goto :end

:end
echo.
echo Press any key to exit...
pause > nul 