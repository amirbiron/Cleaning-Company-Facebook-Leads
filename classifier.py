import os
import json
import openai
from logger import get_logger

log = get_logger("Classifier")

# אתחול lazy — אם אין מפתח, הקליינט ייווצר עם מפתח ריק
# ושגיאה תתפס בזמן קריאת API (לא בזמן import)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-missing"))

# ── רשימת מודלים לפי סדר עדיפות (auto-fallback כשמודל יוצא מתוקף) ──
_MODEL_PRIORITY = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

# מודל פעיל — נטען מ-DB אם שמור, אחרת ברירת מחדל ראשונה ברשימה
_active_model: str | None = None


def _get_active_model() -> str:
    """מחזיר את המודל הפעיל. אם אין — טוען מ-DB או ברירת מחדל."""
    global _active_model
    if _active_model is not None:
        return _active_model
    try:
        from database import get_config
        saved = get_config("openai_model")
        if saved and saved in _MODEL_PRIORITY:
            _active_model = saved
            return _active_model
    except Exception:
        pass
    _active_model = _MODEL_PRIORITY[0]
    return _active_model


def _rotate_model(failed_model: str) -> str | None:
    """מסובב למודל הבא ברשימה אחרי שמודל נכשל (deprecated/not found).
    שומר את המודל החדש ב-DB. מחזיר None אם אין עוד מודלים.
    עוטף מסביב לתחילת הרשימה אם הגענו לסוף (כדי שמודל ישן ב-DB לא יחסום את השרשרת).
    """
    global _active_model
    try:
        idx = _MODEL_PRIORITY.index(failed_model)
    except ValueError:
        idx = -1
    next_idx = (idx + 1) % len(_MODEL_PRIORITY)
    new_model = _MODEL_PRIORITY[next_idx]
    if new_model == failed_model:
        # רק מודל אחד ברשימה — אין לאן לעבור
        log.error("כל המודלים ברשימה נכשלו — אין fallback נוסף")
        return None
    _active_model = new_model
    log.warning(f"מודל {failed_model} לא זמין — עובר ל-{new_model}")
    try:
        from database import set_config
        set_config("openai_model", new_model)
    except Exception:
        pass
    return new_model


def _is_model_deprecated_error(e: openai.APIError) -> bool:
    """בודק אם השגיאה מעידה שהמודל יצא מתוקף או לא קיים."""
    msg = str(e).lower()
    if hasattr(e, "status_code"):
        # 404 = model not found, 410 = gone/deprecated
        if e.status_code in (404, 410):
            return True
    deprecated_hints = ["deprecated", "decommissioned", "does not exist",
                        "model not found", "invalid model", "not available"]
    return any(hint in msg for hint in deprecated_hints)


# קריטריוני סיווג — ברירת מחדל (ניתן לעדכן דרך הפאנל או משתנה סביבה CLASSIFICATION_CRITERIA)
# אם לא הוגדר — ברירת מחדל ריקה (חובה להגדיר דרך env/פאנל לפני הפעלה).
_CLASSIFICATION_CRITERIA_DEFAULT = os.environ.get("CLASSIFICATION_CRITERIA", "").strip()


def _get_classification_criteria() -> str:
    """טוען קריטריוני סיווג מ-DB עם fallback לברירת מחדל."""
    try:
        from database import get_config
        custom = get_config("classification_criteria")
        if custom and custom.strip():
            return custom
    except Exception:
        pass
    return _CLASSIFICATION_CRITERIA_DEFAULT


def _build_system_prompt() -> str:
    return _get_classification_criteria() + """

החזר JSON בלבד, ללא טקסט נוסף:
{"relevant": true/false, "reason": "משפט קצר בעברית"}"""


def _build_batch_system_prompt() -> str:
    return _get_classification_criteria() + """

תקבל מספר פוסטים ממוספרים. החזר מערך JSON בלבד, ללא טקסט נוסף.
כל איבר במערך מתאים לפוסט לפי הסדר:
[{"relevant": true/false, "reason": "משפט קצר בעברית"}, ...]"""


def _chat_completion(system: str, user: str, max_tokens: int,
                     call_type: str = "single") -> str:
    """קריאת Chat Completion עם auto-fallback למודל הבא אם הנוכחי deprecated."""
    model = _get_active_model()
    tried: set[str] = set()
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # שמירת נתוני שימוש (טוקנים ועלות)
            _track_usage(response, model, call_type)
            return (response.choices[0].message.content or "") if response.choices else ""
        except openai.APIError as e:
            if _is_model_deprecated_error(e):
                tried.add(model)
                next_model = _rotate_model(model)
                if next_model is None or next_model in tried:
                    raise  # כל המודלים נוסו — מעלים את השגיאה
                model = next_model
                continue
            raise  # שגיאה אחרת (rate limit, server error) — מעלים


def _track_usage(response, model: str, call_type: str):
    """שומר נתוני שימוש מתשובת API ל-DB."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        from database import save_api_usage
        save_api_usage(model, prompt_tokens, completion_tokens, total_tokens, call_type)
        log.debug(f"[USAGE] {model}: {prompt_tokens}+{completion_tokens}={total_tokens} tokens ({call_type})")
    except Exception as e:
        log.warning(f"שגיאה בשמירת נתוני שימוש: {e}")


def classify_post(content: str, group_name: str) -> dict:
    if not _get_classification_criteria():
        log.error("לא הוגדרו קריטריוני סיווג (CLASSIFICATION_CRITERIA / פאנל) — מדלג על סיווג")
        return {"relevant": False, "reason": "לא הוגדרו קריטריוני סיווג"}
    try:
        log.debug(f"שולח לסיווג: {content[:80]}...")
        raw_text = _chat_completion(
            system=_build_system_prompt(),
            user=f"קבוצה: {group_name}\n\nפוסט:\n{content[:1000]}",
            max_tokens=1024,
            call_type="single",
        )

        log.debug(f"תשובה גולמית מ-API: {raw_text[:200]}")

        if not raw_text.strip():
            log.warning("API החזיר תשובה ריקה")
            return {"relevant": False, "reason": "תשובה ריקה מ-API"}

        result = _parse_json_response(raw_text)
        log.debug(f"תוצאה: relevant={result.get('relevant')} reason={result.get('reason', '')}")
        return result
    except json.JSONDecodeError as e:
        log.error(f"שגיאה בפירוש JSON: {e} | תשובה: {raw_text[:300]}")
        return {"relevant": False, "reason": "שגיאה בפירוש תשובה"}
    except openai.APIError as e:
        status = getattr(e, "status_code", "?")
        log.error(f"שגיאת API: {status} {e}")
        return {"relevant": False, "reason": f"שגיאת API: {status}"}
    except Exception as e:
        log.error(f"שגיאה בסיווג: {e}", exc_info=True)
        return {"relevant": False, "reason": "שגיאה בסיווג"}


def _parse_json_response(raw_text: str):
    """חילוץ JSON (אובייקט או מערך) מתשובת API, כולל עטיפת markdown."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ── סיווג מכרזים ──────────────────────────────────────────────

# קריטריוני סיווג מכרזים — ברירת מחדל מובנית (ניתן לדרוס דרך env/DB)
_TENDER_CRITERIA_DEFAULT_TEXT = """אתה מסווג מכרזים ממשלתיים עבור חברת ניקיון.

מכרז רלוונטי:
- שירותי ניקיון למוסדות ציבוריים, משרדים, מפעלים, בתי אבות, מתנ"סים, דיור מוגן, בתי ספר, שטחים מסחריים
- שירותי תחזוקה ואחזקה שכוללים ניקיון כחלק מרכזי
- שירותי הדברה בשילוב ניקיון
- מכרזים לאספקת ציוד ניקיון בכמויות מוסדיות

מכרז לא רלוונטי:
- ניקיון דירות מגורים, בתים פרטיים, בנייני מגורים
- מכרזים שמזכירים ניקיון רק כחלק שולי (למשל "ניקיון נתונים", "ניקיון תוכנה")
- מכרזי בנייה/שיפוץ שכוללים ניקיון סופי בלבד
- מכרזי כוח אדם כלליים שלא מתמחים בניקיון"""

_TENDER_CRITERIA_ENV = os.environ.get("TENDER_CLASSIFICATION_CRITERIA", "").strip()


def _get_tender_criteria() -> str:
    """טוען קריטריוני סיווג מכרזים מ-DB עם fallback ל-env ולברירת מחדל."""
    try:
        from database import get_config
        custom = get_config("tender_classification_criteria")
        if custom and custom.strip():
            return custom
    except Exception:
        pass
    return _TENDER_CRITERIA_ENV or _TENDER_CRITERIA_DEFAULT_TEXT


def _build_tender_system_prompt() -> str:
    return _get_tender_criteria() + """

תקבל מספר מכרזים ממוספרים. החזר מערך JSON בלבד, ללא טקסט נוסף.
כל איבר במערך מתאים למכרז לפי הסדר:
[{"relevant": true/false, "reason": "משפט קצר בעברית"}, ...]"""


def _build_tender_single_prompt() -> str:
    return _get_tender_criteria() + """

החזר JSON בלבד, ללא טקסט נוסף:
{"relevant": true/false, "reason": "משפט קצר בעברית"}"""


def classify_tender(name: str, publisher: str, category: str) -> dict:
    """סיווג מכרז בודד — האם רלוונטי לחברת ניקיון."""
    criteria = _get_tender_criteria()
    if not criteria:
        return {"relevant": True, "reason": "אין קריטריוני סיווג — מעביר"}
    try:
        raw_text = _chat_completion(
            system=_build_tender_single_prompt(),
            user=f"שם המכרז: {name}\nגוף מפרסם: {publisher}\nתחום: {category}",
            max_tokens=256,
            call_type="tender_single",
        )
        if not raw_text.strip():
            return {"relevant": True, "reason": "תשובה ריקה — מעביר"}
        return _parse_json_response(raw_text)
    except Exception as e:
        log.warning(f"שגיאה בסיווג מכרז: {e} — מעביר")
        return {"relevant": True, "reason": f"שגיאה בסיווג: {e}"}


def classify_tender_batch(tenders: list[dict], batch_size: int = 10) -> list[dict]:
    """סיווג מכרזים באצ'ים — האם רלוונטיים לחברת ניקיון.

    Args:
        tenders: רשימת מילונים עם 'name', 'publisher', 'category'.
        batch_size: כמה מכרזים בכל בקשה (ברירת מחדל: 10 — מכרזים קצרים יותר מפוסטים).

    Returns:
        רשימה באותו סדר, כל איבר {'relevant': bool, 'reason': str}.
    """
    if not tenders:
        return []
    criteria = _get_tender_criteria()
    if not criteria:
        return [{"relevant": True, "reason": "אין קריטריוני סיווג — מעביר"} for _ in tenders]

    all_results: list[dict] = []

    for start in range(0, len(tenders), batch_size):
        batch = tenders[start : start + batch_size]
        log.info(f"סיווג מכרזים באצ' {start // batch_size + 1}: {len(batch)} מכרזים (מתוך {len(tenders)})")

        parts = []
        for idx, t in enumerate(batch, 1):
            parts.append(
                f"--- מכרז {idx} ---\n"
                f"שם: {t.get('name', '')}\n"
                f"גוף מפרסם: {t.get('publisher', '')}\n"
                f"תחום: {t.get('category', '')}"
            )
        user_message = "\n\n".join(parts)

        try:
            raw_text = _chat_completion(
                system=_build_tender_system_prompt(),
                user=user_message,
                max_tokens=max(512, 200 * len(batch)),
                call_type="tender_batch",
            )

            if not raw_text.strip():
                log.warning("תשובה ריקה בבאצ' מכרזים — fallback לסיווג בודד")
                for t in batch:
                    all_results.append(classify_tender(t["name"], t["publisher"], t.get("category", "")))
                continue

            parsed = _parse_json_response(raw_text)
            if isinstance(parsed, dict):
                parsed = [parsed]

            if not isinstance(parsed, list) or len(parsed) != len(batch):
                log.warning(f"אורך תשובה לא תואם בבאצ' מכרזים — fallback לסיווג בודד")
                for t in batch:
                    all_results.append(classify_tender(t["name"], t["publisher"], t.get("category", "")))
                continue

            batch_results = []
            for result in parsed:
                batch_results.append(result)
            all_results.extend(batch_results)

        except json.JSONDecodeError as e:
            log.error(f"JSON לא תקין בבאצ' מכרזים: {e} — fallback")
            for t in batch:
                all_results.append(classify_tender(t["name"], t["publisher"], t.get("category", "")))
        except openai.APIError as e:
            log.error(f"שגיאת API בבאצ' מכרזים: {e} — fallback")
            for t in batch:
                all_results.append(classify_tender(t["name"], t["publisher"], t.get("category", "")))
        except Exception as e:
            log.error(f"שגיאה בסיווג באצ' מכרזים: {e} — fallback", exc_info=True)
            for t in batch:
                all_results.append(classify_tender(t["name"], t["publisher"], t.get("category", "")))

    return all_results


def classify_batch(posts: list[dict], batch_size: int = 5) -> list[dict]:
    """סיווג מספר פוסטים בבקשה אחת ל-API.

    Args:
        posts: רשימת מילונים עם מפתחות 'content' ו-'group'.
        batch_size: כמה פוסטים לשלוח בכל בקשה (ברירת מחדל: 5).

    Returns:
        רשימה באותו סדר, כל איבר {'relevant': bool, 'reason': str}.
    """
    if not posts:
        return []
    if not _get_classification_criteria():
        log.error("לא הוגדרו קריטריוני סיווג (CLASSIFICATION_CRITERIA / פאנל) — מדלג על סיווג")
        return [{"relevant": False, "reason": "לא הוגדרו קריטריוני סיווג"} for _ in posts]

    all_results: list[dict] = []

    for start in range(0, len(posts), batch_size):
        batch = posts[start : start + batch_size]
        log.info(f"סיווג באצ' {start // batch_size + 1}: {len(batch)} פוסטים (מתוך {len(posts)})")

        # בניית prompt מרובה פוסטים
        parts = []
        for idx, post in enumerate(batch, 1):
            content = post.get("content", "")[:1000]
            group = post.get("group", "")
            parts.append(f"--- פוסט {idx} ---\nקבוצה: {group}\n\n{content}")
        user_message = "\n\n".join(parts)

        try:
            raw_text = _chat_completion(
                system=_build_batch_system_prompt(),
                user=user_message,
                max_tokens=max(1024, 500 * len(batch)),
                call_type="batch",
            )

            log.debug(f"תשובה באצ': {raw_text[:300]}")

            if not raw_text.strip():
                log.warning("API החזיר תשובה ריקה בבאצ' — fallback לסיווג בודד")
                for post in batch:
                    all_results.append(classify_post(post["content"], post["group"]))
                continue

            parsed = _parse_json_response(raw_text)

            # אם API החזיר אובייקט בודד במקום מערך (באצ' של 1)
            if isinstance(parsed, dict):
                parsed = [parsed]

            if not isinstance(parsed, list) or len(parsed) != len(batch):
                log.warning(
                    f"אורך תשובה ({len(parsed) if isinstance(parsed, list) else 'N/A'}) "
                    f"לא תואם לבאצ' ({len(batch)}) — fallback לסיווג בודד"
                )
                for post in batch:
                    all_results.append(classify_post(post["content"], post["group"]))
                continue

            # אוספים תוצאות לרשימה זמנית — אם אלמנט לא תקין, ה-exception
            # ייתפס לפני שמוסיפים תוצאות חלקיות ל-all_results
            batch_results = []
            for idx, result in enumerate(parsed):
                log.debug(
                    f"פוסט {start + idx + 1}: "
                    f"relevant={result.get('relevant')} reason={result.get('reason', '')}"
                )
                batch_results.append(result)
            all_results.extend(batch_results)

        except json.JSONDecodeError as e:
            log.error(f"שגיאה בפירוש JSON בבאצ': {e} — fallback לסיווג בודד")
            for post in batch:
                all_results.append(classify_post(post["content"], post["group"]))
        except openai.APIError as e:
            status = getattr(e, "status_code", "?")
            log.error(f"שגיאת API בבאצ': {status} {e} — fallback לסיווג בודד")
            for post in batch:
                all_results.append(classify_post(post["content"], post["group"]))
        except Exception as e:
            log.error(f"שגיאה בסיווג באצ': {e} — fallback לסיווג בודד", exc_info=True)
            for post in batch:
                all_results.append(classify_post(post["content"], post["group"]))

    return all_results
