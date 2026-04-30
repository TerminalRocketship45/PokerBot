@echo off
cd /d C:\Users\rohan\Downloads\ML\RohanPoker
call C:\Users\rohan\anaconda3FINAL\Scripts\activate.bat C:\Users\rohan\anaconda3FINAL\envs\rl_env
echo Starting poker server...
echo Open your browser at: http://localhost:5000
python src\ui\web_app.py --checkpoint checkpoints\hunl_final.pt
pause
