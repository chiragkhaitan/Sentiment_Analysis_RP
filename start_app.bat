@echo off
echo ==========================================
echo Starting Twitter Sentiment Analysis App
echo ==========================================
echo.

echo Checking if models exist...
if not exist "models\vectoriser.pickle" (
    echo ⚠ ERROR: Models not found!
    echo Please run 'setup.bat' first to train the models.
    echo.
    pause
    exit /b 1
)

echo ✓ Models found!
echo.
echo Starting Flask server...
echo.
echo Open your browser and go to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
