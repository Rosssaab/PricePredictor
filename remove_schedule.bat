@echo off
echo Removing Price Predictor Schedule...

:: Remove scheduled task
SCHTASKS /DELETE /TN "CryptoAI\PricePredictor" /F 2>nul

echo.
echo Price Predictor schedule has been removed.
echo.
pause 