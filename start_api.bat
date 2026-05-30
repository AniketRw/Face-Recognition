@echo off

cd /d "D:\Aniket\Face Recognition"

"C:\Program Files\Python310\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000

pause