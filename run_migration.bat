@echo off
echo Running migration: VK support...
call .venv\Scripts\activate
python migrate_vk_support.py
pause
