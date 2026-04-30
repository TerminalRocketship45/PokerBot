@echo off
cd /d C:\Users\rohan\Downloads\ML\RohanPoker
call C:\Users\rohan\anaconda3FINAL\Scripts\activate.bat C:\Users\rohan\anaconda3FINAL\envs\rl_env
echo Starting medium training (500 iter, warm-start from iter_0100)...
python scripts\run_training.py --config medium --bc_checkpoint checkpoints\hunl_iter_0100.pt
echo Done.
pause
