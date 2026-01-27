## Telegram Mini App — Полёт к звёздам

### Установка

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Создай файл `.env` рядом с `main.py`:

```env
BOT_TOKEN=ваш_бот_токен_от_BotFather
BOT_USERNAME=имя_бота_без_@
CHANNEL_USERNAME=photavel
DB_PATH=db.sqlite3
```

### Запуск сервера

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

После этого фронтенд (`index.html`) будет доступен по адресу `http://127.0.0.1:8000/` и может использоваться как Telegram WebApp.

