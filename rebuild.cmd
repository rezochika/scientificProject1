@echo off
timeout 5
echo.
cd "C:\Users\rezoc\PycharmProjects\scientificProject1"

start /wait "" "C:\Users\rezoc\anaconda3\envs\tf-gpu-test\python.exe" "BuildModels.py"
timeout 5