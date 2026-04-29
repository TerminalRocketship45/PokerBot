@echo off
cd /d C:\Users\rohan\Downloads\ML\RohanPoker
call C:\Users\rohan\anaconda3FINAL\Scripts\activate.bat C:\Users\rohan\anaconda3FINAL\envs\rl_env
python scripts\run_training.py --config quick
