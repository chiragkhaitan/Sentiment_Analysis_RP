@echo off
echo ==========================================
echo Twitter Sentiment Analysis - Setup Script
echo ==========================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed successfully!
echo.

echo Step 2: Checking for dataset...
if exist "training.1600000.processed.noemoticon.csv" (
    echo ✓ Dataset found!
) else (
    echo ⚠ WARNING: Dataset not found!
    echo Please download the dataset from:
    echo https://www.kaggle.com/datasets/kazanova/sentiment140
    echo.
    echo Extract 'training.1600000.processed.noemoticon.csv' to this folder
    echo.
    pause
    exit /b 1
)
echo.

echo Step 3: Training models (this may take 10-30 minutes)...
echo Please be patient...
python train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo.

echo ==========================================
echo ✓ Setup completed successfully!
echo ==========================================
echo.
echo To start the web application, run: start_app.bat
echo Or manually run: python app.py
echo.
pause
