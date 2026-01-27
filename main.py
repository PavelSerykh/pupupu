import os
import hmac
import hashlib
import sqlite3
import secrets
from datetime import datetime, timedelta, date
from urllib.parse import parse_qsl

import httpx
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Конфиг / окружение ---
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
BOT_USERNAME = os.getenv("BOT_USERNAME", "your_bot_username")  # без @
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "photavel")   # без @
DB_PATH = os.getenv("DB_PATH", "db.sqlite3")

if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
    print("!! ВНИМАНИЕ: укажите BOT_TOKEN в .env")

# --- Инициализация FastAPI ---
app = FastAPI()

# --- CORS (разрешаем все, можно ужесточить при желании) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании укажи конкретный origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Работа с БД (sqlite3, минималистично) ---

def get_db():
    """Открываем подключение к SQLite на запрос."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Создаём таблицы при старте."""
    conn = get_db()
    cur = conn.cursor()
    # Таблица пользователей
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE NOT NULL,
            username TEXT,
            balance INTEGER NOT NULL DEFAULT 0,
            is_verified INTEGER NOT NULL DEFAULT 0,
            ref_code TEXT UNIQUE,
            referred_by INTEGER,
            ref_bonus_given INTEGER NOT NULL DEFAULT 0,
            last_game_date TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


# --- Модели Pydantic для запросов/ответов ---


class ProfileOut(BaseModel):
    username: str | None
    balance: int
    is_verified: bool
    ref_link: str
    can_play_today: bool


class CheckSubOut(BaseModel):
    is_subscribed: bool
    new_balance: int
    is_verified: bool


class GameStatusOut(BaseModel):
    can_play_today: bool
    seconds: int


class GameFinishIn(BaseModel):
    stars_collected: int


class GameFinishOut(BaseModel):
    added: int
    new_balance: int
    next_available: str


class SpinOut(BaseModel):
    prize_type: str
    prize_stars: int
    new_balance: int


class CatalogItem(BaseModel):
    id: int
    title: str
    description: str


class InitAuthIn(BaseModel):
    initData: str
    start_param: str | None = None


# --- Вспомогательные функции для Telegram WebApp auth ---


def get_telegram_secret_key(bot_token: str) -> bytes:
    """Секретный ключ = SHA256(bot_token)."""
    return hashlib.sha256(bot_token.encode()).digest()


def verify_init_data(init_data: str, bot_token: str) -> dict:
    """
    Проверяем подпись initData по документации Telegram.
    Возвращаем словарь всех параметров (включая user, start_param).
    """
    parsed = dict(parse_qsl(init_data, keep_blank_values=True))
    hash_received = parsed.pop("hash", None)
    if not hash_received:
        raise HTTPException(status_code=401, detail="No hash in initData")

    data_check_string = "\n".join(
        sorted(f"{k}={v}" for k, v in parsed.items())
    )

    secret_key = get_telegram_secret_key(bot_token)
    calc_hash = hmac.new(
        secret_key,
        msg=data_check_string.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(calc_hash, hash_received):
        raise HTTPException(status_code=401, detail="Invalid initData hash")

    return parsed


class CurrentUser(BaseModel):
    id: int
    telegram_id: int
    username: str | None
    balance: int
    is_verified: bool
    ref_code: str


def get_or_create_user(
    conn: sqlite3.Connection,
    telegram_id: int,
    username: str | None,
    start_param: str | None,
) -> sqlite3.Row:
    """
    Создаём пользователя при первом заходе.
    Обрабатываем реферальную ссылку (start_param с префиксом ref_).
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
    row = cur.fetchone()
    if row:
        if username and row["username"] != username:
            cur.execute("UPDATE users SET username = ? WHERE id = ?", (username, row["id"]))
            conn.commit()
        return row

    ref_code = secrets.token_urlsafe(6)
    created_at = datetime.utcnow().isoformat()
    referred_by_id = None

    if start_param and start_param.startswith("ref_"):
        inviter_code = start_param[4:]
        cur.execute("SELECT id FROM users WHERE ref_code = ?", (inviter_code,))
        inviter = cur.fetchone()
        if inviter:
            referred_by_id = inviter["id"]

    cur.execute(
        """
        INSERT INTO users (telegram_id, username, balance, is_verified,
                           ref_code, referred_by, ref_bonus_given,
                           last_game_date, created_at)
        VALUES (?, ?, 0, 0, ?, ?, 0, NULL, ?)
        """,
        (telegram_id, username, ref_code, referred_by_id, created_at),
    )
    conn.commit()
    cur.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
    return cur.fetchone()


async def get_current_user(
    x_telegram_initdata: str = Header(..., alias="X-Telegram-InitData"),
    x_telegram_user_id: int = Header(..., alias="X-Telegram-User-Id"),
    x_telegram_username: str | None = Header(None, alias="X-Telegram-Username"),
):
    """
    Достаём пользователя на основе initData + user_id из заголовка.
    Фронт обязан передать:
    - X-Telegram-InitData: window.Telegram.WebApp.initData
    - X-Telegram-User-Id: user.id (из initDataUnsafe.user)
    - X-Telegram-Username: user.username (если есть)
    """
    verify_init_data(x_telegram_initdata, BOT_TOKEN)

    conn = get_db()
    try:
        parsed = dict(parse_qsl(x_telegram_initdata, keep_blank_values=True))
        start_param = parsed.get("start_param")
        user_row = get_or_create_user(conn, x_telegram_user_id, x_telegram_username, start_param)
    finally:
        conn.close()

    return CurrentUser(
        id=user_row["id"],
        telegram_id=user_row["telegram_id"],
        username=user_row["username"],
        balance=user_row["balance"],
        is_verified=bool(user_row["is_verified"]),
        ref_code=user_row["ref_code"],
    )


# --- Маршрут для index.html ---


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.post("/api/auth/init", response_model=ProfileOut)
async def auth_init(data: InitAuthIn):
    raise HTTPException(status_code=400, detail="Используйте заголовки X-Telegram-* для авторизации")


@app.get("/api/me", response_model=ProfileOut)
async def get_me(user: CurrentUser = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE id = ?", (user.id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        last_game_date = row["last_game_date"]
        can_play_today = True
        if last_game_date:
            try:
                d = date.fromisoformat(last_game_date)
                can_play_today = (d != date.today())
            except ValueError:
                can_play_today = True

        ref_link = f"https://t.me/{BOT_USERNAME}?start=ref_{row['ref_code']}"

        return ProfileOut(
            username=row["username"],
            balance=row["balance"],
            is_verified=bool(row["is_verified"]),
            ref_link=ref_link,
            can_play_today=can_play_today,
        )
    finally:
        conn.close()


async def check_subscription_via_bot(telegram_id: int) -> bool:
    """Проверяем подписку через Bot API getChatMember."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getChatMember"
    chat_id = f"@{CHANNEL_USERNAME}"
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(url, params={"chat_id": chat_id, "user_id": telegram_id})
    data = resp.json()
    if not data.get("ok"):
        return False
    status = data["result"]["status"]
    return status in ["member", "administrator", "creator"]


@app.post("/api/check-subscription", response_model=CheckSubOut)
async def check_subscription(user: CurrentUser = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor()
    try:
        is_sub = await check_subscription_via_bot(user.telegram_id)
        if not is_sub:
            return CheckSubOut(is_subscribed=False, new_balance=user.balance, is_verified=False)

        cur.execute("SELECT * FROM users WHERE id = ?", (user.id,))
        row = cur.fetchone()
        if row and row["is_verified"]:
            return CheckSubOut(
                is_subscribed=True,
                new_balance=row["balance"],
                is_verified=True,
            )

        new_balance = row["balance"]

        if row["referred_by"] and not row["ref_bonus_given"]:
            cur.execute("UPDATE users SET ref_bonus_given = 1 WHERE id = ?", (row["id"],))
            cur.execute(
                "UPDATE users SET balance = balance + 30 WHERE id = ?",
                (row["referred_by"],),
            )

        cur.execute(
            "UPDATE users SET is_verified = 1 WHERE id = ?",
            (user.id,),
        )
        conn.commit()

        cur.execute("SELECT balance FROM users WHERE id = ?", (user.id,))
        new_balance = cur.fetchone()["balance"]

        return CheckSubOut(
            is_subscribed=True,
            new_balance=new_balance,
            is_verified=True,
        )
    finally:
        conn.close()


@app.get("/api/game/status", response_model=GameStatusOut)
async def game_status(user: CurrentUser = Depends(get_current_user)):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT last_game_date FROM users WHERE id = ?", (user.id,))
        row = cur.fetchone()
        last_game_date = row["last_game_date"] if row else None

        if not last_game_date:
            return GameStatusOut(can_play_today=True, seconds=30)
        try:
            d = date.fromisoformat(last_game_date)
        except ValueError:
            return GameStatusOut(can_play_today=True, seconds=30)

        if d == date.today():
            return GameStatusOut(can_play_today=False, seconds=30)
        else:
            return GameStatusOut(can_play_today=True, seconds=30)
    finally:
        conn.close()


@app.post("/api/game/finish", response_model=GameFinishOut)
async def game_finish(
    payload: GameFinishIn,
    user: CurrentUser = Depends(get_current_user),
):
    if payload.stars_collected < 0:
        raise HTTPException(status_code=400, detail="Invalid stars count")

    stars_to_add = min(payload.stars_collected, 20)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT last_game_date, balance FROM users WHERE id = ?", (user.id,))
        row = cur.fetchone()
        last_game_date = row["last_game_date"]
        if last_game_date:
            try:
                d = date.fromisoformat(last_game_date)
                if d == date.today():
                    raise HTTPException(status_code=400, detail="Already played today")
            except ValueError:
                pass

        new_balance = row["balance"] + stars_to_add
        cur.execute(
            "UPDATE users SET balance = ?, last_game_date = ? WHERE id = ?",
            (new_balance, date.today().isoformat(), user.id),
        )
        conn.commit()

        next_available = (date.today() + timedelta(days=1)).isoformat()
        return GameFinishOut(
            added=stars_to_add,
            new_balance=new_balance,
            next_available=next_available,
        )
    finally:
        conn.close()


@app.post("/api/roulette/spin", response_model=SpinOut)
async def roulette_spin(user: CurrentUser = Depends(get_current_user)):
    """Рулетка: 1 прокрутка = 10 звёзд."""
    import random

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT balance FROM users WHERE id = ?", (user.id,))
        row = cur.fetchone()
        balance = row["balance"]
        if balance < 10:
            raise HTTPException(status_code=400, detail="Not enough stars (need 10)")

        balance -= 10

        prizes = [
            ("stars", 5, 40),
            ("stars", 10, 30),
            ("stars", 20, 15),
            ("stars", 30, 8),
            ("stars", 40, 5),
            ("stars", 50, 1),
            ("iphone", 0, 1),
        ]
        weights = [p[2] for p in prizes]
        choice = random.choices(prizes, weights=weights, k=1)[0]
        prize_type, prize_stars, _ = choice

        balance += prize_stars

        cur.execute(
            "UPDATE users SET balance = ? WHERE id = ?",
            (balance, user.id),
        )
        conn.commit()

        return SpinOut(
            prize_type=prize_type,
            prize_stars=prize_stars,
            new_balance=balance,
        )
    finally:
        conn.close()


@app.get("/api/catalog", response_model=list[CatalogItem])
async def catalog(user: CurrentUser = Depends(get_current_user)):
    """Статический каталог из 9 заглушек."""
    items = [
        CatalogItem(id=i, title=f"Товар {i}", description="Описание товара-заглушки")
        for i in range(1, 10)
    ]
    return items

