import asyncio
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message

TOKEN = "7677104461:AAHGXYcEr4AcoWn7zbBRBQ3S2NM_WW2a85g"

router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("✅ Бот работает! Это тестовый ответ.")

@router.message(F.text)
async def echo(message: Message):
    await message.answer(f"Получил: {message.text}")

async def main():
    bot = Bot(token=TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    print("🤖 Тестовый бот запущен...")
    print("Попробуйте написать /start в Telegram")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
