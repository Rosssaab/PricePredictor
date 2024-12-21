@echo off
echo Setting up Price Predictor Schedule...

:: Create the task with multiple triggers (every 6 hours)
SCHTASKS /CREATE /TN "CryptoAI\PricePredictor" /TR "C:\PythonApps\DataLoaders\FillPredictions\run_predictor.bat" /SC HOURLY /MO 6 /ST 00:00 /F

:: Verify the task was created
SCHTASKS /QUERY /TN "CryptoAI\PricePredictor"

echo.
echo Price Predictor has been scheduled to run every 6 hours.
echo Starting times: 00:00, 06:00, 12:00, 18:00
echo.
pause 