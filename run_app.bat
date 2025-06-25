@echo off
echo 🚀 Launching Ghibli Recipe Garden with performance optimizations...
echo.

REM Activate the conda environment if it exists
if exist "recipe-env" (
    echo 📦 Activating conda environment...
    call conda activate recipe-env
) else (
    echo ⚠️  Conda environment not found, using system Python...
)

REM Set environment variables for better performance
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

REM Launch streamlit with optimizations
echo 🌿 Starting the app...
python -m streamlit run app.py --server.port 8501 --server.address localhost --server.headless true --browser.gatherUsageStats false

echo.
echo �� App stopped.
pause 