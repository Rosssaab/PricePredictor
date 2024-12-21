@echo off
echo Starting Price Predictor...

:: Set the working directory (change this to your actual path)
cd /d C:\PythonApps\DataLoaders\FillPredictions

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the predictor
python PricePredictor.py --debug >> logs\predictor_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1

:: Deactivate virtual environment
call deactivate

echo Price Predictor finished. 