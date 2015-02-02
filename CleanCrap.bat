@echo off
echo Cleaning Stuff ........

rmdir Debug /s /q
rmdir Release /s /q
rmdir LibISR\Debug /s /q
rmdir LibISR\Release /s /q
rmdir LibISRUtils\Debug /s /q
rmdir LibISRUtils\Release /s /q
rmdir ORUtils\Debug /s /q
rmdir ORUtils\Release /s /q
rmdir UI\Debug /s /q
rmdir UI\Release /s /q
del /f /s /q *.sdf

echo Cleaning Done!
echo. & pause