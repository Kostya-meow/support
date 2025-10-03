import requests
import sys

# Токен из .env файла
TOKEN = "7677104461:AAHGXYcEr4AcoWn7zbBRBQ3S2NM_WW2a85g"

# Проверяем webhook
response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo")
data = response.json()

print("=== Webhook Info ===")
print(f"Success: {data.get('ok')}")
if data.get('ok'):
    result = data.get('result', {})
    webhook_url = result.get('url', '')
    print(f"Webhook URL: {webhook_url if webhook_url else 'NOT SET (good for polling)'}")
    print(f"Has custom certificate: {result.get('has_custom_certificate')}")
    print(f"Pending update count: {result.get('pending_update_count')}")
    
    if webhook_url:
        print("\n❌ ПРОБЛЕМА: Webhook установлен! Нужно удалить его.")
        print("Удаляю webhook...")
        delete_response = requests.get(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook")
        if delete_response.json().get('ok'):
            print("✅ Webhook удален! Теперь бот будет работать в режиме polling.")
        else:
            print("❌ Не удалось удалить webhook")
    else:
        print("\n✅ Webhook не установлен, бот должен работать в режиме polling")

# Проверяем валидность токена
me_response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe")
me_data = me_response.json()

print("\n=== Bot Info ===")
if me_data.get('ok'):
    bot_info = me_data.get('result', {})
    print(f"✅ Токен валидный!")
    print(f"Bot ID: {bot_info.get('id')}")
    print(f"Bot Username: @{bot_info.get('username')}")
    print(f"Bot Name: {bot_info.get('first_name')}")
else:
    print(f"❌ Токен невалидный: {me_data}")
