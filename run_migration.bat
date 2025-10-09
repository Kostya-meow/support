@echo off
echo Running migration: add is_read field...
call .venv\Scripts\activate
python migrate_add_is_read.py
pause
