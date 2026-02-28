#!/usr/bin/env python3
"""
Security News Bot
RSS → Groq (краткое резюме на русском) → gTTS → RVC (Каневский) → Telegram
"""

import os
import time
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests
from groq import Groq
from gtts import gTTS

# ─── Конфигурация ────────────────────────────────────────────────────────────

FEEDS = {
    "🔴 BleepingComputer":  "https://www.bleepingcomputer.com/feed/",
    "🔴 The Hacker News":   "https://feeds.feedburner.com/TheHackersNews",
    "🔴 Kaspersky Blog":    "https://www.kaspersky.com/blog/feed/",
    "🔴 BlockThreat":       "https://blockthreat.io/feed/",
    "🔴 Objective-See":     "https://objective-see.org/rss.xml",
}

MAX_ARTICLES_PER_BLOG = 2  # максимум статей с одного блога
LOOKBACK_HOURS        = 24  # смотрим за последние N часов
MODEL_CACHE_DIR       = Path.home() / ".rvc_models" / "kanevsky"

# Из GitHub Secrets
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
TG_TOKEN        = os.environ["TELEGRAM_TOKEN"]
TG_CHAT_ID      = os.environ["TELEGRAM_CHAT_ID"]
HF_MODEL_URL    = os.environ.get("HF_MODEL_URL", "")  # URL zip с моделью


# ─── RSS: получение новых статей ─────────────────────────────────────────────

def get_recent_articles(feed_url: str) -> list[dict]:
    """Возвращает статьи опубликованные за последние LOOKBACK_HOURS часов."""
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"  [RSS] Ошибка парсинга {feed_url}: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    articles = []

    for entry in feed.entries:
        pub = None
        for attr in ("published_parsed", "updated_parsed"):
            val = getattr(entry, attr, None)
            if val:
                pub = datetime(*val[:6], tzinfo=timezone.utc)
                break

        if pub is None or pub < cutoff:
            continue

        articles.append({
            "title":   entry.get("title", "Без заголовка").strip(),
            "link":    entry.get("link", ""),
            "content": entry.get("summary", entry.get("description", ""))[:4000],
            "pub":     pub,
        })

        if len(articles) >= MAX_ARTICLES_PER_BLOG:
            break

    return articles


# ─── Groq: краткое резюме на русском ─────────────────────────────────────────

def summarize_to_russian(title: str, content: str) -> str:
    """3 предложения по-русски — кратко и понятно."""
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "Ты — диктор новостей кибербезопасности. "
        "Напиши краткое резюме этой статьи НА РУССКОМ языке — ровно 3 предложения. "
        "Простым языком, без технического жаргона, без вступлений вроде 'Вот резюме:'. "
        "Только сам текст резюме.\n\n"
        f"Заголовок: {title}\n\n"
        f"Текст: {content}"
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Groq] Ошибка: {e}")
        return f"Новая статья: {title}"


# ─── gTTS: текст → mp3 ───────────────────────────────────────────────────────

def tts_to_mp3(text: str, out_path: str):
    """Синтез речи на русском через Google TTS."""
    tts = gTTS(text=text, lang="ru", slow=False)
    tts.save(out_path)


# ─── RVC: смена голоса на Каневского ─────────────────────────────────────────

def download_model_if_needed() -> tuple[str, str]:
    """
    Скачивает ZIP с моделью (один раз, затем кэш).
    Возвращает (путь к .pth, путь к .index или '').
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pth_files   = list(MODEL_CACHE_DIR.glob("*.pth"))
    index_files = list(MODEL_CACHE_DIR.glob("*.index"))

    if pth_files:
        print(f"  [RVC] Модель в кэше: {pth_files[0]}")
        return str(pth_files[0]), str(index_files[0]) if index_files else ""

    print(f"  [RVC] Скачиваю модель с {HF_MODEL_URL}...")
    zip_path = MODEL_CACHE_DIR / "model.zip"

    r = requests.get(HF_MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(MODEL_CACHE_DIR)
    zip_path.unlink()

    pth_files   = list(MODEL_CACHE_DIR.rglob("*.pth"))
    index_files = list(MODEL_CACHE_DIR.rglob("*.index"))

    if not pth_files:
        raise FileNotFoundError("В ZIP не найден .pth файл модели")

    return str(pth_files[0]), str(index_files[0]) if index_files else ""


def apply_rvc(mp3_in: str, mp3_out: str, model_pth: str, model_index: str):
    """mp3 → wav → RVC (Каневский) → mp3."""
    from rvc_python.infer import RVCInference

    # mp3 → wav через ffmpeg
    wav_in  = mp3_in.replace(".mp3", "_in.wav")
    wav_out = mp3_in.replace(".mp3", "_rvc.wav")

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_in, wav_in],
        check=True, capture_output=True
    )

    # RVC inference
    rvc = RVCInference(device="cpu")
    rvc.load_model(model_pth)
    if model_index:
        rvc.index_path = model_index
    rvc.infer_file(wav_in, wav_out)

    # wav → mp3
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_out, "-codec:a", "libmp3lame", "-qscale:a", "4", mp3_out],
        check=True, capture_output=True
    )

    # Чистим временные wav
    for f in [wav_in, wav_out]:
        if os.path.exists(f):
            os.unlink(f)


# ─── Telegram ────────────────────────────────────────────────────────────────

def tg_send_text(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=30)


def tg_send_audio(title: str, link: str, audio_path: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendAudio"
    caption = f"<b>{title}</b>\n\n🔗 <a href='{link}'>Читать полностью</a>"

    with open(audio_path, "rb") as f:
        resp = requests.post(url, data={
            "chat_id": TG_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML",
        }, files={"audio": ("news.mp3", f, "audio/mpeg")}, timeout=60)

    if not resp.ok:
        print(f"  [TG] Ошибка отправки аудио: {resp.text[:200]}")
        # fallback: отправить хотя бы текст
        tg_send_text(f"📄 <b>{title}</b>\n\n🔗 {link}")


# ─── Главный цикл ────────────────────────────────────────────────────────────

def main():
    print(f"=== Security News Bot — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    # Загрузить модель RVC один раз (если настроена)
    use_rvc    = bool(HF_MODEL_URL)
    model_pth  = ""
    model_index = ""

    if use_rvc:
        try:
            model_pth, model_index = download_model_if_needed()
            print(f"[RVC] Модель готова: {model_pth}\n")
        except Exception as e:
            print(f"[RVC] Не удалось загрузить модель: {e} → используем gTTS без RVC\n")
            use_rvc = False

    total_sent = 0

    for blog_name, feed_url in FEEDS.items():
        print(f"📰 {blog_name}")
        articles = get_recent_articles(feed_url)

        if not articles:
            print("  Новых статей нет\n")
            continue

        # Заголовок блога в Telegram
        date_str = datetime.now(timezone.utc).strftime("%d.%m.%Y")
        tg_send_text(
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{blog_name}\n"
            f"📅 {date_str} · {len(articles)} {'статья' if len(articles) == 1 else 'статьи'}"
        )
        time.sleep(1)

        for article in articles:
            title   = article["title"]
            link    = article["link"]
            content = article["content"]
            print(f"  → {title[:70]}...")

            # 1. Groq: краткое резюме
            summary = summarize_to_russian(title, content)
            print(f"     Резюме: {summary[:80]}...")

            # 2. TTS (mp3)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts_path = tmp.name

            tts_to_mp3(f"{title}. {summary}", tts_path)

            # 3. RVC (опционально)
            audio_path = tts_path
            if use_rvc:
                rvc_path = tts_path.replace(".mp3", "_kanevsky.mp3")
                try:
                    apply_rvc(tts_path, rvc_path, model_pth, model_index)
                    audio_path = rvc_path
                    print("     RVC ✓")
                except Exception as e:
                    print(f"     RVC ошибка: {e} → gTTS")

            # 4. Отправить в Telegram
            tg_send_audio(title, link, audio_path)
            total_sent += 1
            print("     Отправлено в Telegram ✓")

            # Чистим файлы
            for f in [tts_path, audio_path]:
                if os.path.exists(f):
                    os.unlink(f)

            time.sleep(2)  # пауза между сообщениями

        print()

    if total_sent == 0:
        print("Новых статей сегодня не найдено — бот молчит.")
    else:
        print(f"\n✅ Готово. Отправлено статей: {total_sent}")


if __name__ == "__main__":
    main()
