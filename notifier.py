import os
import re
import requests
from logger import get_logger

log = get_logger("Notifier")

def _get_bot_token() -> str | None:
    return os.environ.get("TELEGRAM_BOT_TOKEN")

def _get_chat_id() -> str | None:
    return os.environ.get("TELEGRAM_CHAT_ID")

# backward compat — שומרים את השמות הישנים למי שקורא אותם ישירות
BOT_TOKEN = _get_bot_token()
CHAT_ID = _get_chat_id()
# chat ID נפרד להתראות שגיאה — אם מוגדר, שגיאות ילכו למפתח במקום ללקוח
ERROR_CHAT_ID = os.environ.get("ERROR_CHAT_ID")

def send_message(
    text: str,
    *,
    chat_id: str | int | None = None,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> bool:
    bot_token = _get_bot_token()
    default_chat = _get_chat_id()
    if not bot_token or not (chat_id or default_chat):
        log.error("חסרים TELEGRAM_BOT_TOKEN או TELEGRAM_CHAT_ID")
        return False

    target_chat = str(chat_id or default_chat)

    payload: dict = {
        "chat_id": target_chat,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        log.debug(f"שולח הודעה לטלגרם (chat_id={target_chat})...")
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            log.info("הודעה נשלחה בהצלחה לטלגרם")
            return True

        log.error(f"טלגרם החזיר סטטוס {response.status_code}: {response.text}")
        return False
    except Exception as e:
        log.error(f"שגיאה בשליחה לטלגרם: {e}", exc_info=True)
        return False

def clean_post_content(text: str) -> str:
    """מנקה תוכן פוסט מ-Unicode פרטי של פייסבוק ורעשי UI."""
    # הסרת תגיות HTML שדלפו (כולל <img src="data:..."> עם base64)
    text = re.sub(r'<[^>]+>', '', text)

    # הסרת תווים מ-Private Use Areas (U+E000–U+F8FF, U+F0000–U+10FFFF)
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    text = re.sub(r'[\U000F0000-\U0010FFFF]', '', text)

    # הסרת שורות רעש של פייסבוק (לייקים, תגובות, שיתופים)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # דילוג על שורות שהן רק מספרים (ספירת לייקים/תגובות)
        if re.match(r'^\d+$', stripped):
            continue
        # דילוג על "ו-עוד X אחרים" (רשימת מגיבים)
        if re.search(r'ו-עוד \d+ אחרי', stripped):
            continue
        # דילוג על שורות ריקות לגמרי (אחרי ניקוי Unicode)
        if not stripped:
            cleaned.append('')
            continue
        cleaned.append(line)

    text = '\n'.join(cleaned)
    # ניקוי שורות ריקות מיותרות שנשארו
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def send_message_with_buttons(
    text: str,
    buttons: list[list[dict]],
    *,
    chat_id: str | int | None = None,
    parse_mode: str | None = None,
    disable_web_page_preview: bool = False,
) -> dict | None:
    """שולח הודעה עם InlineKeyboard.
    buttons — מערך דו-ממדי של כפתורים, כל dict: {"text": "...", "callback_data": "..."}.
    מחזיר dict של ההודעה שנשלחה (כולל message_id) או None בשגיאה.
    """
    bot_token = _get_bot_token()
    default_chat = _get_chat_id()
    if not bot_token or not (chat_id or default_chat):
        log.error("חסרים TELEGRAM_BOT_TOKEN או TELEGRAM_CHAT_ID")
        return None

    target_chat = str(chat_id or default_chat)

    payload: dict = {
        "chat_id": target_chat,
        "text": text,
        "reply_markup": {"inline_keyboard": buttons},
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        log.error(f"טלגרם החזיר סטטוס {response.status_code}: {response.text}")
        return None
    except Exception as e:
        log.error(f"שגיאה בשליחת הודעה עם כפתורים: {e}", exc_info=True)
        return None


def edit_message_text(
    chat_id: int | str,
    message_id: int,
    text: str,
    buttons: list[list[dict]] | None = None,
    *,
    parse_mode: str | None = None,
) -> bool:
    """עורך הודעה קיימת (טקסט + כפתורים אופציונליים)."""
    bot_token = _get_bot_token()
    if not bot_token:
        return False

    payload: dict = {
        "chat_id": str(chat_id),
        "message_id": message_id,
        "text": text,
    }
    if buttons is not None:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/editMessageText",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            return True
        log.error(f"editMessageText נכשל: {response.status_code}: {response.text}")
        return False
    except Exception as e:
        log.error(f"שגיאה בעריכת הודעה: {e}", exc_info=True)
        return False


def answer_callback_query(callback_query_id: str, text: str = "") -> bool:
    """עונה ל-callback query (מסיר את אנימציית הטעינה מהכפתור)."""
    bot_token = _get_bot_token()
    if not bot_token:
        return False
    payload: dict = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/answerCallbackQuery",
            json=payload,
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def send_document(
    file_bytes: bytes,
    filename: str,
    caption: str = "",
    *,
    chat_id: str | int | None = None,
) -> bool:
    """שולח קובץ לטלגרם דרך sendDocument.

    file_bytes — תוכן הקובץ כ-bytes.
    filename — שם הקובץ שיוצג בטלגרם.
    caption — כיתוב אופציונלי (עד 1024 תווים).
    """
    bot_token = _get_bot_token()
    default_chat = _get_chat_id()
    if not bot_token or not (chat_id or default_chat):
        log.error("חסרים TELEGRAM_BOT_TOKEN או TELEGRAM_CHAT_ID")
        return False

    target_chat = str(chat_id or default_chat)

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendDocument",
            data={"chat_id": target_chat, "caption": caption[:1024]},
            files={"document": (filename, file_bytes)},
            timeout=15,
        )
        if response.status_code == 200:
            log.debug(f"קובץ {filename} נשלח לטלגרם")
            return True
        log.error(f"sendDocument נכשל: {response.status_code}: {response.text}")
        return False
    except Exception as e:
        log.error(f"שגיאה בשליחת קובץ לטלגרם: {e}", exc_info=True)
        return False


def send_error_alert(error: str) -> bool:
    """שולח שגיאה קריטית כהודעת טלגרם (חתוך ל-500 תווים).
    אם ERROR_CHAT_ID מוגדר — שולח למפתח. אחרת — ל-CHAT_ID הרגיל.
    """
    truncated = error[:500]
    return send_message(
        f"⚠️ שגיאה בבוט:\n{truncated}",
        chat_id=ERROR_CHAT_ID,
        disable_web_page_preview=True,
    )


def send_tender(name: str, publisher: str, deadline: str, url: str,
                category: str = "", source: str = "mr.gov.il"):
    """שולח הודעת מכרז חדש לטלגרם."""
    if not _get_bot_token() or not _get_chat_id():
        log.error("חסרים משתני סביבה של טלגרם (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False

    deadline_line = f"\n\u23f0 *מועד אחרון:* {deadline}" if deadline else ""
    publisher_line = f"\n\U0001f3e2 *גוף מפרסם:* {publisher}" if publisher else ""
    category_line = f"\n\U0001f4c2 *תחום:* {category}" if category else ""

    text = (
        f"\U0001f4cb *מכרז חדש*"
        f"{publisher_line}"
        f"\n\n{name}"
        f"{category_line}"
        f"{deadline_line}"
        f"\n\n\U0001f517 [פתח מכרז]({url})"
        f"\n\U0001f4cd מקור: {source}"
    )
    return send_message(
        text,
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


def _to_desktop_url(url: str) -> str:
    """ממיר URL מובייל ל-URL דסקטופ (m.facebook.com → www.facebook.com)."""
    return url.replace("://m.facebook.com", "://www.facebook.com")

def send_lead(group_name: str, content: str, post_url: str, reason: str,
              has_real_url: bool = True, also_in: list[str] | None = None,
              is_hot: bool = False, author_url: str = ""):
    if not _get_bot_token() or not _get_chat_id():
        log.error("חסרים משתני סביבה של טלגרם (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False

    content = clean_post_content(content)
    short_content = content[:350] + "..." if len(content) > 350 else content

    # המרת URL מובייל לדסקטופ — נוח יותר למשתמשים
    desktop_url = _to_desktop_url(post_url)

    # טקסט הלינק משתנה לפי סוג ה-URL — פוסט ספציפי או קבוצה בלבד
    link_label = "פתח פוסט" if has_real_url else "פתח קבוצה"

    # תצוגת קבוצה — אם הפוסט הופיע גם בקבוצות נוספות, מציג אותן
    group_line = f"\U0001f4cc *קבוצה:* {group_name}"
    if also_in:
        group_line += f" (+ {len(also_in)} קבוצות נוספות)"

    # שורת ליד חם — מופיעה בראש ההודעה אם הפוסט מכיל מילה חמה
    hot_line = ""
    if is_hot:
        hot_line = "\U0001f6a8 ליד חם! \U0001f6a8\n\n"

    # שורת פרופיל — מוצגת רק אם יש קישור לפרופיל המחבר
    profile_line = ""
    if author_url:
        desktop_author = _to_desktop_url(author_url)
        profile_line = f"\n\U0001f464 [פרופיל המפרסם]({desktop_author})"

    text = (
        f"{hot_line}"
        f"\U0001f3af *ליד חדש*\n"
        f"{group_line}\n\n"
        f"{short_content}\n\n"
        f"\U0001f4a1 *למה רלוונטי:* {reason}\n\n"
        f"\U0001f517 [{link_label}]({desktop_url})"
        f"{profile_line}"
    )
    return send_message(
        text,
        parse_mode="Markdown",
        disable_web_page_preview=False,
    )
