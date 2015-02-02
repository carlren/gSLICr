@echo off
echo Cleaning Stuff ........

rmdir Debug /s /q
rmdir Release /s /q
del /f /s /q *.sdf

echo Cleaning Done!
echo. & pause