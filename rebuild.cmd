@echo off
set mydate=%date:~10,4%%date:~4,2%%date:~7,2%
echo %mydate%
timeout 5
echo.
cd "C:\Users\rezoc\PycharmProjects\scientificProject1"


start /wait "" "C:\Users\rezoc\anaconda3\envs\tf-gpu-test\python.exe" "dnnlstm.py" "rebuild=1" "modelPath=modelLSTM%mydate%/"
timeout 5
start /wait "" "C:\Users\rezoc\anaconda3\envs\tf-gpu-test\python.exe" "dnnlstm.py" "rebuild=1" "bi=1" "modelPath=modelLSTMBidirectional%mydate%/"