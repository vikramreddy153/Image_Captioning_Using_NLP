@echo off
echo ================================
echo Setting up virtual environment...
echo ================================
cd /d D:\Image_Captioning_NLP

:: Delete old venv if it exists
IF EXIST venv (
    echo Removing old venv...
    rmdir /s /q venv
)

:: Create new venv
python -m venv venv

:: Activate venv and install packages
call venv\Scripts\activate

echo ================================
echo Upgrading pip...
echo ================================
python -m ensurepip --upgrade
python -m pip install --upgrade pip

echo ================================
echo Installing Streamlit...
echo ================================
pip install streamlit==1.38.0

echo ================================
echo ✅ All done!
echo Run: venv\Scripts\activate
echo Then: streamlit run app.py
echo ================================
pause
