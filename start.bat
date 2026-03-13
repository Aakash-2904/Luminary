@echo off
echo Starting Luminary...

start cmd /k "cd C:\Users\uttka\Luminary\backend && python -m uvicorn main:app --reload --port 8000"

timeout /t 3

start cmd /k "cd C:\Users\uttka\Luminary\frontend && npm run dev"

echo Both servers starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
pause