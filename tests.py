"""Unit tests for Facebook-Leads-New — issue #28."""

import json
import re
from datetime import datetime, time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# 1. passes_keyword_filter
# ──────────────────────────────────────────────────────────────

from main import passes_keyword_filter


_TEST_PRE_FILTER = ["בוט", "אוטומציה", "סקריפט", "מתכנת", "פיתוח"]


class TestPassesKeywordFilter:
    def test_matching_keyword(self):
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("אני צריך בוט לטלגרם") is True

    def test_no_match(self):
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("שלום לכולם, מה קורה?") is False

    def test_case_insensitive(self):
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("אוטומציה") is True

    def test_empty_string(self):
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("") is False

    def test_keyword_as_substring(self):
        # "מתכנת" appears inside a longer word — should still match
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("אני מתכנתת מנוסה") is True

    def test_multiple_keywords(self):
        with patch.dict("main._keywords_state", {"pre_filter": _TEST_PRE_FILTER}):
            assert passes_keyword_filter("פיתוח בוט אוטומציה") is True

    def test_empty_list_passes_all(self):
        """רשימת מילות מפתח ריקה = אין סינון מוקדם, הכל עובר."""
        with patch.dict("main._keywords_state", {"pre_filter": []}):
            assert passes_keyword_filter("כל טקסט שהוא") is True


# ──────────────────────────────────────────────────────────────
# 2. is_blocked
# ──────────────────────────────────────────────────────────────

from main import is_blocked


class TestIsBlocked:
    def test_no_block_keywords(self):
        with patch.dict("main._keywords_state", {"block": []}):
            assert is_blocked("כל טקסט שהוא") is False

    def test_blocked_word_present(self):
        with patch.dict("main._keywords_state", {"block": ["ספאם", "פרסום"]}):
            assert is_blocked("זה ספאם מלא") is True

    def test_blocked_word_absent(self):
        with patch.dict("main._keywords_state", {"block": ["ספאם"]}):
            assert is_blocked("פוסט רגיל לגמרי") is False

    def test_case_insensitive(self):
        with patch.dict("main._keywords_state", {"block": ["spam"]}):
            assert is_blocked("This is SPAM") is True


# ──────────────────────────────────────────────────────────────
# 2b. matches_force_send
# ──────────────────────────────────────────────────────────────

from main import matches_force_send


class TestForceSend:
    def test_no_force_keywords(self):
        with patch.dict("main._keywords_state", {"force_send": [], "group_force_send": {}}):
            assert matches_force_send("כל טקסט שהוא") is None

    def test_match_returns_keyword(self):
        with patch.dict("main._keywords_state", {"force_send": ["skill", "getdrip"], "group_force_send": {}}):
            assert matches_force_send("פוסט על Skill כלשהו") == "skill"

    def test_no_match_returns_none(self):
        with patch.dict("main._keywords_state", {"force_send": ["skill"], "group_force_send": {}}):
            assert matches_force_send("פוסט רגיל") is None

    def test_case_insensitive(self):
        with patch.dict("main._keywords_state", {"force_send": ["skill"], "group_force_send": {}}):
            assert matches_force_send("SKILL is great") == "skill"

    def test_multiple_keywords_first_match(self):
        with patch.dict("main._keywords_state", {"force_send": ["skill", "getdrip"], "group_force_send": {}}):
            result = matches_force_send("GetDRIP הוא כלי מעולה")
            assert result == "getdrip"


class TestForceSendCRUD:
    def test_add_and_load(self, monkeypatch, tmp_path):
        """הוספת מילה ל-force_send ושליפה."""
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        database.init_db()
        ok, msg = add_force_send_keyword("skill")
        assert ok
        keywords = _load_force_send_keywords()
        assert "skill" in keywords

    def test_add_duplicate(self, monkeypatch, tmp_path):
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        database.init_db()
        add_force_send_keyword("skill")
        ok, msg = add_force_send_keyword("skill")
        assert not ok

    def test_remove(self, monkeypatch, tmp_path):
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        database.init_db()
        add_force_send_keyword("skill")
        ok, msg = remove_force_send_keyword("skill")
        assert ok
        assert "skill" not in _load_force_send_keywords()

    def test_remove_not_found(self, monkeypatch, tmp_path):
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        database.init_db()
        ok, msg = remove_force_send_keyword("nonexistent")
        assert not ok

    def test_lowercased(self, monkeypatch, tmp_path):
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        database.init_db()
        add_force_send_keyword("SKILL")
        assert "skill" in _load_force_send_keywords()


from main import add_force_send_keyword, remove_force_send_keyword, _load_force_send_keywords


# ──────────────────────────────────────────────────────────────
# 3. _parse_quiet_hours / _is_quiet_now
# ──────────────────────────────────────────────────────────────

from main import _parse_quiet_hours, _is_quiet_now


class TestParseQuietHours:
    def test_standard_format(self):
        result = _parse_quiet_hours("22:00-06:00")
        assert result == (time(22, 0), time(6, 0))

    def test_short_format(self):
        result = _parse_quiet_hours("2-7")
        assert result == (time(2, 0), time(7, 0))

    def test_empty_string(self):
        assert _parse_quiet_hours("") is None

    def test_none_input(self):
        assert _parse_quiet_hours(None) is None

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            _parse_quiet_hours("invalid")

    def test_whitespace(self):
        result = _parse_quiet_hours("  02:00 - 07:00  ")
        assert result == (time(2, 0), time(7, 0))


class TestIsQuietNow:
    def test_within_quiet_hours(self):
        quiet = (time(22, 0), time(6, 0))
        now = datetime(2026, 1, 1, 23, 30)
        assert _is_quiet_now(now, quiet) is True

    def test_outside_quiet_hours(self):
        quiet = (time(22, 0), time(6, 0))
        now = datetime(2026, 1, 1, 12, 0)
        assert _is_quiet_now(now, quiet) is False

    def test_before_midnight_wrap(self):
        quiet = (time(22, 0), time(6, 0))
        now = datetime(2026, 1, 1, 5, 59)
        assert _is_quiet_now(now, quiet) is True

    def test_at_end_boundary(self):
        quiet = (time(22, 0), time(6, 0))
        now = datetime(2026, 1, 1, 6, 0)
        assert _is_quiet_now(now, quiet) is False

    def test_same_start_end(self):
        quiet = (time(0, 0), time(0, 0))
        now = datetime(2026, 1, 1, 15, 0)
        assert _is_quiet_now(now, quiet) is True

    def test_no_midnight_wrap(self):
        quiet = (time(2, 0), time(7, 0))
        now = datetime(2026, 1, 1, 3, 0)
        assert _is_quiet_now(now, quiet) is True

    def test_no_midnight_wrap_outside(self):
        quiet = (time(2, 0), time(7, 0))
        now = datetime(2026, 1, 1, 8, 0)
        assert _is_quiet_now(now, quiet) is False


# ──────────────────────────────────────────────────────────────
# 4. clean_post_content
# ──────────────────────────────────────────────────────────────

from notifier import clean_post_content, send_error_alert, send_document


class TestCleanPostContent:
    def test_removes_private_use_unicode(self):
        text = "שלום \uE000\uE001 עולם"
        assert "\uE000" not in clean_post_content(text)

    def test_removes_like_count_lines(self):
        text = "פוסט מעניין\n42\nתגובה"
        result = clean_post_content(text)
        assert "42" not in result

    def test_removes_hebrew_commenter_lines(self):
        text = "תוכן\nו-עוד 5 אחרי\nסוף"
        result = clean_post_content(text)
        assert "ו-עוד" not in result

    def test_collapses_excess_newlines(self):
        text = "שורה 1\n\n\n\n\nשורה 2"
        result = clean_post_content(text)
        assert "\n\n\n" not in result

    def test_strips_whitespace(self):
        assert clean_post_content("  שלום  ") == "שלום"

    def test_normal_text_unchanged(self):
        text = "מחפש מפתח לבניית בוט"
        assert clean_post_content(text) == text

    def test_strips_html_img_with_base64(self):
        text = 'תוכן <img src="data:image/jpg;base64,AAAA"> עוד טקסט'
        result = clean_post_content(text)
        assert "<img" not in result
        assert "base64" not in result
        assert "תוכן" in result
        assert "עוד טקסט" in result

    def test_strips_generic_html_tags(self):
        text = "שלום <b>עולם</b> <span>טקסט</span>"
        result = clean_post_content(text)
        assert "<b>" not in result
        assert "<span>" not in result
        assert "שלום עולם טקסט" == result


# ──────────────────────────────────────────────────────────────
# 4b. send_error_alert
# ──────────────────────────────────────────────────────────────


class TestSendErrorAlert:
    @patch("notifier.send_message")
    def test_sends_error_message(self, mock_send):
        mock_send.return_value = True
        result = send_error_alert("something broke")
        assert result is True
        mock_send.assert_called_once()
        call_text = mock_send.call_args[0][0]
        assert "⚠️" in call_text
        assert "something broke" in call_text

    @patch("notifier.send_message")
    def test_truncates_long_error(self, mock_send):
        mock_send.return_value = True
        long_error = "x" * 1000
        send_error_alert(long_error)
        call_text = mock_send.call_args[0][0]
        # הודעת השגיאה חתוכה ל-500 תווים מהשגיאה עצמה
        assert len(call_text) < 600

    @patch("notifier.send_message")
    def test_send_failure_returns_false(self, mock_send):
        mock_send.return_value = False
        result = send_error_alert("error")
        assert result is False

    @patch("notifier.send_message")
    def test_sends_to_error_chat_id_when_set(self, mock_send):
        """כש-ERROR_CHAT_ID מוגדר, שגיאות נשלחות למפתח."""
        mock_send.return_value = True
        import notifier
        original = notifier.ERROR_CHAT_ID
        try:
            notifier.ERROR_CHAT_ID = "999999"
            send_error_alert("test error")
            assert mock_send.call_args[1]["chat_id"] == "999999"
        finally:
            notifier.ERROR_CHAT_ID = original

    @patch("notifier.send_message")
    def test_falls_back_to_default_when_error_chat_id_unset(self, mock_send):
        """כש-ERROR_CHAT_ID לא מוגדר, שגיאות נשלחות ל-CHAT_ID הרגיל."""
        mock_send.return_value = True
        import notifier
        original = notifier.ERROR_CHAT_ID
        try:
            notifier.ERROR_CHAT_ID = None
            send_error_alert("test error")
            # chat_id=None → send_message ישתמש ב-CHAT_ID הרגיל
            assert mock_send.call_args[1]["chat_id"] is None
        finally:
            notifier.ERROR_CHAT_ID = original


# ──────────────────────────────────────────────────────────────
# 4c. send_document
# ──────────────────────────────────────────────────────────────


class TestSendDocument:
    @patch("notifier.requests.post")
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier._get_chat_id", new=lambda: "123")
    def test_sends_file_successfully(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        result = send_document(b"<html>test</html>", "dump.html", "caption")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "sendDocument" in call_kwargs[0][0]
        assert call_kwargs[1]["data"]["chat_id"] == "123"
        assert call_kwargs[1]["data"]["caption"] == "caption"

    @patch("notifier.requests.post")
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier._get_chat_id", new=lambda: "123")
    def test_returns_false_on_failure(self, mock_post):
        mock_post.return_value = MagicMock(status_code=400, text="Bad request")
        result = send_document(b"<html>test</html>", "dump.html")
        assert result is False

    @patch("notifier.requests.post")
    @patch("notifier._get_bot_token", new=lambda: None)
    @patch("notifier._get_chat_id", new=lambda: "123")
    def test_returns_false_without_token(self, mock_post):
        result = send_document(b"<html>test</html>", "dump.html")
        assert result is False
        mock_post.assert_not_called()

    @patch("notifier.requests.post")
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier._get_chat_id", new=lambda: "123")
    def test_truncates_caption_to_1024(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        long_caption = "x" * 2000
        send_document(b"data", "file.html", long_caption)
        call_kwargs = mock_post.call_args
        assert len(call_kwargs[1]["data"]["caption"]) == 1024

    @patch("notifier.requests.post")
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier._get_chat_id", new=lambda: "123")
    def test_custom_chat_id(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        send_document(b"data", "file.html", chat_id="999")
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["chat_id"] == "999"


# ──────────────────────────────────────────────────────────────
# 5. extract_post_id
# ──────────────────────────────────────────────────────────────

from scraper import extract_post_id


class TestExtractPostId:
    def test_posts_pattern(self):
        assert extract_post_id("https://facebook.com/groups/123/posts/456789") == "456789"

    def test_story_fbid_pattern(self):
        assert extract_post_id("https://facebook.com/story.php?story_fbid=123456&id=789") == "123456"

    def test_permalink_pattern(self):
        assert extract_post_id("https://facebook.com/groups/123/permalink/999888") == "999888"

    def test_pfbid_pattern(self):
        url = "https://facebook.com/groups/123/posts/pfbidABCdef123"
        result = extract_post_id(url)
        assert result == "ABCdef123"

    def test_share_p_pattern(self):
        """פורמט share link חדש של פייסבוק — /share/p/ID."""
        assert extract_post_id("https://www.facebook.com/share/p/1AkvNbuYNi/") == "1AkvNbuYNi"

    def test_share_p_mobile(self):
        """share link מאתר מובייל."""
        assert extract_post_id("https://m.facebook.com/share/p/Xb2cD3fG/") == "Xb2cD3fG"

    def test_fallback_last_20_chars(self):
        url = "https://facebook.com/some/unknown/format"
        result = extract_post_id(url)
        assert result == url[-20:]

    def test_short_url_fallback(self):
        url = "https://fb.com/x"
        result = extract_post_id(url)
        assert result == url[-20:]


# ──────────────────────────────────────────────────────────────
# 5b. content-based post ID for posts without real URL
# ──────────────────────────────────────────────────────────────

import hashlib


class TestContentBasedPostId:
    """פוסטים ללא URL אמיתי צריכים לקבל ID מבוסס hash תוכן,
    כדי למנוע ID זהה לכל הפוסטים בקבוצה (באג #55)."""

    def test_different_content_different_id(self):
        """שני פוסטים עם תוכן שונה מקבלים ID שונה."""
        text1 = "פוסט ראשון על נושא מסוים"
        text2 = "פוסט שני על נושא אחר"
        id1 = "c_" + hashlib.md5(text1.encode()).hexdigest()[:16]
        id2 = "c_" + hashlib.md5(text2.encode()).hexdigest()[:16]
        assert id1 != id2

    def test_same_content_same_id(self):
        """אותו תוכן מייצר אותו ID — dedup עובד כראוי."""
        text = "פוסט חוזר עם אותו תוכן בדיוק"
        id1 = "c_" + hashlib.md5(text.encode()).hexdigest()[:16]
        id2 = "c_" + hashlib.md5(text.encode()).hexdigest()[:16]
        assert id1 == id2

    def test_content_id_has_prefix(self):
        """ID מבוסס תוכן מתחיל ב-'c_' כדי להבדיל מ-ID רגיל."""
        text = "תוכן כלשהו"
        post_id = "c_" + hashlib.md5(text.encode()).hexdigest()[:16]
        assert post_id.startswith("c_")
        assert len(post_id) == 18  # "c_" + 16 hex chars

    def test_content_id_not_equal_to_group_fallback(self):
        """ID מבוסס תוכן שונה מ-fallback של URL קבוצה."""
        group_url = "https://m.facebook.com/groups/640894321475593/"
        text = "פוסט כלשהו בקבוצה הזאת"
        fallback_id = extract_post_id(group_url)
        content_id = "c_" + hashlib.md5(text.encode()).hexdigest()[:16]
        assert content_id != fallback_id


# ──────────────────────────────────────────────────────────────
# 5b. _stable_text_for_hash — נרמול טקסט ליצירת hash יציב
# ──────────────────────────────────────────────────────────────

from scraper import _stable_text_for_hash


class TestStableTextForHash:
    """ה-hash לפוסטים ללא URL אמיתי חייב להיות יציב בין סריקות —
    timestamps, לייקים ורעשי UI לא משפיעים."""

    def test_timestamp_hebrew_removed(self):
        """timestamps בעברית מוסרים."""
        t1 = "מחפש דירה בתל אביב\nלפני 2 שעות"
        t2 = "מחפש דירה בתל אביב\nלפני 5 שעות"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_timestamp_hebrew_yamim_removed(self):
        """'לפני 3 ימים' (plural days) מוסר — ימים מתחיל ב-ימ, לא יו."""
        t1 = "מחפש דירה בתל אביב\nלפני 3 ימים"
        t2 = "מחפש דירה בתל אביב\nלפני 7 ימים"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_timestamp_english_removed(self):
        """timestamps באנגלית מוסרים."""
        t1 = "Looking for apartment\n3h"
        t2 = "Looking for apartment\n12h"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_like_count_lines_removed(self):
        """שורות שהן רק מספרים (ספירת לייקים) מוסרות."""
        t1 = "פוסט על עבודה\n5\n3"
        t2 = "פוסט על עבודה\n12\n8"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_ui_strings_removed(self):
        """מחרוזות UI (Like, Comment, Share) מוסרות."""
        t1 = "תוכן הפוסט\nLike\nComment\nShare"
        t2 = "תוכן הפוסט"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_hebrew_ui_strings_removed(self):
        """מחרוזות UI בעברית מוסרות."""
        t1 = "תוכן הפוסט\nהגב\nשתף\nאהבתי"
        t2 = "תוכן הפוסט"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_actual_content_preserved(self):
        """תוכן אמיתי נשמר — פוסטים שונים מייצרים hash שונה."""
        t1 = "מחפש דירה בתל אביב 3 חדרים"
        t2 = "מחפש דירה בירושלים 4 חדרים"
        assert _stable_text_for_hash(t1) != _stable_text_for_hash(t2)

    def test_same_post_different_timestamps_same_hash(self):
        """אותו פוסט עם timestamps שונים מייצר אותו hash."""
        base = "מישהו מכיר חשמלאי טוב באזור השרון? צריך לטפל בתקלה בלוח חשמל"
        t1 = f"{base}\nלפני 2 שעות\n5\nLike\nComment"
        t2 = f"{base}\nלפני 6 שעות\n12\nLike\nComment"
        h1 = hashlib.md5(_stable_text_for_hash(t1).encode()).hexdigest()[:16]
        h2 = hashlib.md5(_stable_text_for_hash(t2).encode()).hexdigest()[:16]
        assert h1 == h2

    def test_yesterday_removed(self):
        """'אתמול' ו-'Yesterday' מוסרים."""
        t1 = "פוסט כלשהו\nאתמול"
        t2 = "פוסט כלשהו\nYesterday"
        t3 = "פוסט כלשהו"
        s1 = _stable_text_for_hash(t1)
        s2 = _stable_text_for_hash(t2)
        s3 = _stable_text_for_hash(t3)
        assert s1 == s2 == s3

    def test_time_unit_mid_sentence_stripped_for_stability(self):
        """מספר + יחידת זמן (inline) מוסר — tradeoff: hash יציב > hash ספציפי.
        "5 שעות" ו-"3 שעות" נמחקים כי הם בפורמט זהה ל-timestamps של פייסבוק.
        הסיכון (collision בין "עבודה של 5 שעות" ל-"עבודה של 3 שעות") נמוך מאוד
        לעומת כפילויות חוזרות שנגרמות מ-hash לא יציב."""
        t1 = "מחפש עבודה של 5 שעות ביום"
        t2 = "מחפש עבודה של 3 שעות ביום"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_standalone_words_mid_sentence_preserved(self):
        """'אתמול' באמצע משפט לא נמחק — תוכן אמיתי."""
        t1 = "אתמול הלכתי לקניון"
        assert "אתמול" in _stable_text_for_hash(t1)

    # ── בדיקות engagement metrics ──────────────────────────────

    def test_engagement_comments_hebrew_removed(self):
        """שורת engagement '5 תגובות' מוסרת."""
        t1 = "מחפש עבודה\n5 תגובות"
        t2 = "מחפש עבודה\n12 תגובות"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)
        assert _stable_text_for_hash(t1) == _stable_text_for_hash("מחפש עבודה")

    def test_engagement_single_comment_removed(self):
        """שורת engagement '1 תגובה' (יחיד) מוסרת."""
        t1 = "מחפש עבודה\n1 תגובה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash("מחפש עבודה")

    def test_engagement_shares_hebrew_removed(self):
        """שורת engagement '3 שיתופים' מוסרת."""
        t1 = "מחפש עבודה\n3 שיתופים"
        t2 = "מחפש עבודה\n10 שיתופים"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_engagement_likes_hebrew_removed(self):
        """שורת engagement '7 לייקים' מוסרת."""
        t1 = "מחפש עבודה\n7 לייקים"
        t2 = "מחפש עבודה\n20 לייקים"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_engagement_english_removed(self):
        """שורות engagement באנגלית מוסרות."""
        base = "Looking for a developer"
        t1 = f"{base}\n5 comments"
        t2 = f"{base}\n12 shares"
        t3 = f"{base}\n3 likes"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(base)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(base)
        assert _stable_text_for_hash(t3) == _stable_text_for_hash(base)

    def test_engagement_combined_line_removed(self):
        """שורת engagement משולבת '5 תגובות · 3 שיתופים' מוסרת."""
        base = "מחפש עבודה"
        t1 = f"{base}\n5 תגובות · 3 שיתופים"
        t2 = f"{base}\n12 תגובות · 7 שיתופים"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(base)

    def test_engagement_mid_sentence_preserved(self):
        """'5 תגובות' באמצע משפט לא נמחק — תוכן אמיתי."""
        t1 = "קיבלתי 5 תגובות על הפוסט"
        assert "5 תגובות" in _stable_text_for_hash(t1)

    def test_engagement_views_removed(self):
        """שורת engagement '100 צפיות' מוסרת."""
        t1 = "מחפש עבודה\n100 צפיות"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash("מחפש עבודה")

    def test_full_post_with_engagement_stable(self):
        """פוסט מלא עם engagement שונה מייצר אותו hash."""
        base = "מישהו מכיר חשמלאי טוב באזור השרון? צריך לטפל בתקלה בלוח חשמל"
        t1 = f"ישראל ישראלי\n{base}\nלפני 2 שעות\n5 תגובות · 3 שיתופים\n5\nLike\nComment"
        t2 = f"ישראל ישראלי\n{base}\nלפני 8 שעות\n12 תגובות · 7 שיתופים\n20\nLike\nComment"
        h1 = hashlib.md5(_stable_text_for_hash(t1).encode()).hexdigest()[:16]
        h2 = hashlib.md5(_stable_text_for_hash(t2).encode()).hexdigest()[:16]
        assert h1 == h2

    # ── נרמול עמוק — URLs, unicode, case ──────────────────────

    def test_url_removed_from_hash(self):
        """URLs עם פרמטרי מעקב שונים מייצרים אותו hash."""
        t1 = "מחפש עבודה\nhttps://example.com/job?fbclid=abc123"
        t2 = "מחפש עבודה\nhttps://example.com/job?fbclid=xyz789"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)

    def test_case_normalized(self):
        """הבדלי case לא משנים את ה-hash."""
        t1 = "Looking For A Developer"
        t2 = "looking for a developer"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_unicode_direction_marks_removed(self):
        """תווי כיווניות (LRM/RLM) לא משפיעים על ה-hash."""
        t1 = "\u200eמחפש עבודה\u200f"
        t2 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_zero_width_chars_removed(self):
        """תווי zero-width לא משפיעים על ה-hash."""
        t1 = "מחפש\u200bעבודה\u2060טובה"
        t2 = "מחפשעבודהטובה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_see_more_removed(self):
        """'קרא עוד' / 'See more' מוסרים."""
        t1 = "מחפש עבודה\nקרא עוד"
        t2 = "מחפש עבודה\nSee more"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_see_translation_removed(self):
        """'ראה תרגום' / 'See translation' מוסרים."""
        t1 = "מחפש עבודה\nראה תרגום"
        t2 = "מחפש עבודה\nSee translation"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_write_comment_removed(self):
        """'כתוב תגובה...' מוסר."""
        t1 = "מחפש עבודה\nכתוב תגובה..."
        t2 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_sponsored_removed(self):
        """'ממומן' / 'Sponsored' מוסרים."""
        t1 = "מחפש עבודה\nממומן"
        t2 = "מחפש עבודה\nSponsored"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_pua_chars_removed(self):
        """תווי Private Use Area (אייקונים של פייסבוק) מוסרים."""
        t1 = "מחפש עבודה\uE001\uE100"
        t2 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_url_in_mid_sentence_removed(self):
        """URL באמצע משפט מוסר ללא פגיעה בתוכן."""
        t1 = "ראו כאן https://fb.com/p/123?ref=share מעניין מאוד"
        t2 = "ראו כאן מעניין מאוד"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_real_content_not_affected_by_normalization(self):
        """תוכן אמיתי לא נפגע — Hebrew content still distinguishes posts."""
        t1 = "מחפש מפתח פייתון לפרויקט"
        t2 = "מחפש מעצב גרפי לפרויקט"
        assert _stable_text_for_hash(t1) != _stable_text_for_hash(t2)

    def test_phone_number_preserved_in_hash(self):
        """מספרי טלפון ישראליים לא נמחקים מה-hash — הם תוכן משמעותי."""
        assert "054-1234567" in _stable_text_for_hash("054-1234567")
        assert "972-52-1234567" in _stable_text_for_hash("+972-52-1234567")
        # פוסטים עם מספרי טלפון שונים חייבים לקבל hash שונה
        t1 = "מחפש עבודה\n054-1234567"
        t2 = "מחפש עבודה\n052-9876543"
        assert _stable_text_for_hash(t1) != _stable_text_for_hash(t2)

    def test_pure_numeric_noise_still_stripped(self):
        """שורות שהן רק מספרים (לייקים/תגובות) עדיין נמחקות."""
        assert _stable_text_for_hash("1,234") == ""
        assert _stable_text_for_hash("5") == ""
        assert _stable_text_for_hash("12.5") == ""

    def test_inline_relative_time_hebrew_stripped(self):
        """זמנים יחסיים בעברית ('8 שעות') מוסרים גם inline — מטא-דאטה של פייסבוק."""
        # הפורמט מהלוג: "שם 8 שעות [אייקונים] תוכן..."
        t1 = "אמיר בירון 8 שעות היי חברים אני מפתח בוט"
        t2 = "אמיר בירון 12 שעות היי חברים אני מפתח בוט"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_inline_relative_time_variants(self):
        """וריאנטים שונים של יחידות זמן בעברית מוסרים."""
        base = "משה כהן %s פוסט חשוב"
        assert _stable_text_for_hash(base % "3 דקות") == _stable_text_for_hash(base % "15 דקות")
        assert _stable_text_for_hash(base % "1 שעה") == _stable_text_for_hash(base % "5 שעות")
        assert _stable_text_for_hash(base % "2 ימים") == _stable_text_for_hash(base % "1 יום")
        assert _stable_text_for_hash(base % "1 שבוע") == _stable_text_for_hash(base % "3 שבועות")

    def test_inline_relative_time_english_stripped(self):
        """English relative times stripped inline."""
        t1 = "john doe 2 hours hey everyone"
        t2 = "john doe 5 hours hey everyone"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)
        t3 = "john doe 30 mins hey everyone"
        t4 = "john doe 45 mins hey everyone"
        assert _stable_text_for_hash(t3) == _stable_text_for_hash(t4)

    def test_time_word_boundary_no_partial_match(self):
        """word boundary מונע התאמה חלקית — 'monthly' ו-'hourly' לא נפגעים."""
        assert "monthly" in _stable_text_for_hash("5 monthly payments")
        assert "hourly" in _stable_text_for_hash("hourly rate 100")
        assert "weekly" in _stable_text_for_hash("3 weekly meetings")
        # "יומי" (daily) לא נפגע
        assert "יומי" in _stable_text_for_hash("שכר יומי 500")

    def test_time_no_cross_line_match(self):
        """מספר בסוף שורה + יחידת זמן בתחילת שורה הבאה — לא נמחקים."""
        # "מחיר 50\nשעות עבודה" — ה-50 ו-"שעות" לא צריכים להתאים
        result = _stable_text_for_hash("מחיר 50\nשעות עבודה")
        assert "50" in result
        assert "שעות עבודה" in result

    def test_years_hebrew_stripped(self):
        """שנים/שנה מוסרים — כמו שאר יחידות הזמן."""
        t1 = "משה כהן 2 שנים פוסט חשוב"
        t2 = "משה כהן 5 שנים פוסט חשוב"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_supplementary_pua_chars_removed(self):
        """אייקוני פרטיות של פייסבוק (supplementary PUA) מוסרים מה-hash."""
        # Supplementary PUA-A (U+F0000-U+FFFFD) and PUA-B (U+100000-U+10FFFD)
        t1 = "אמיר בירון\U000F0001\U00100001 היי חברים"
        t2 = "אמיר בירון היי חברים"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_yesterday_today_stripped(self):
        """'אתמול', 'היום' standalone מוסרים."""
        assert _stable_text_for_hash("אתמול") == ""
        assert _stable_text_for_hash("היום") == ""
        # "אתמול ב-15:30" מוסר inline
        t1 = "משה כהן אתמול ב-15:30 פוסט חשוב"
        t2 = "משה כהן פוסט חשוב"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    # ── שורות דינמיות — "+N", "ו-עוד N אחרים", "עוד" ──────────

    def test_plus_n_reaction_count_removed(self):
        """'+3' — ספירת ריאקשנים/תגובות — שורה שלמה בלבד."""
        t1 = "מחפש עבודה\n+3"
        t2 = "מחפש עבודה\n+7"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_plus_n_in_phone_preserved(self):
        """'+972-52-123' — פלוס במספר טלפון לא נמחק (שורה לא רק '+N')."""
        t1 = "+972-52-1234567"
        assert "+972-52-1234567" in _stable_text_for_hash(t1)

    def test_and_more_others_hebrew_removed(self):
        """'ו-עוד 5 אחרים' — שורת מגיבים מוסרת."""
        t1 = "מחפש עבודה\nו-עוד 5 אחרים"
        t2 = "מחפש עבודה\nו-עוד 12 אחרים"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_and_more_english_removed(self):
        """'and 3 others' / 'and 5 more' — removed."""
        t1 = "post content\nand 3 others"
        t2 = "post content\nand 5 more"
        t3 = "post content"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_standalone_od_removed(self):
        """'עוד' ו-'... עוד' standalone מוסרים — See more וריאנט."""
        t1 = "מחפש עבודה\nעוד"
        t2 = "מחפש עבודה\n... עוד"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_most_relevant_hebrew_removed(self):
        """'הרלוונטיות ביותר' / 'החדשות ביותר' — מיון תגובות UI מוסר."""
        t1 = "מחפש עבודה\nהרלוונטיות ביותר"
        t2 = "מחפש עבודה\nהחדשות ביותר"
        t3 = "מחפש עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_most_relevant_english_removed(self):
        """'Most relevant' / 'Newest' — comment sorting UI removed."""
        t1 = "post content\nMost relevant"
        t2 = "post content\nNewest"
        t3 = "post content"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    # ── core hash: עמידות בפני תגובות שמתווספות בין סריקות ────

    def test_core_hash_same_post_with_comments_added(self):
        """אותו פוסט, אבל תגובות חדשות התווספו — core hash זהה."""
        from main import _content_dedup_hash
        base_post = (
            "Neta Levy\n7 שעות\n"
            "היי, אני נטע וזה ניסיון ראשון לבארטר מהסוג הזה🙂\n"
            "אני צריכה דף נחיתה לסדנאות שלי ומישהו שיודע\n"
            "לעשות קמפיין פרסום ממומן בפייסבוק\n"
            "אני מציעה בתמורה ליווי אישי + תפריט תזונתי\n"
        )
        # סריקה 1 — ללא תגובות
        t1 = base_post + "+3\n3 תגובות\nאהבתי\nהגב\nשתף"
        # סריקה 2 — שעה אחרי, עם תגובות
        t2 = (base_post.replace("7 שעות", "8 שעות")
              + "+5\n5 תגובות · 2 שיתופים\nאהבתי\nהגב\nשתף\n"
              + "הרלוונטיות ביותר\n"
              + "David Cohen\nשלחי לי הודעה, אני יכול לעזור\n"
              + "אהבתי\nהגב\n"
              + "Sarah Levi\nגם אני מעוניינת!\nאהבתי\nהגב\n"
              + "כתוב תגובה...")
        h1 = _content_dedup_hash(t1)
        h2 = _content_dedup_hash(t2)
        assert h1 == h2, (
            f"core hash צריך להיות זהה גם כשתגובות מתווספות: "
            f"{h1} != {h2}"
        )

    def test_core_hash_different_posts_stay_different(self):
        """פוסטים שונים לגמרי — core hash שונה."""
        from main import _content_dedup_hash
        t1 = "נטע לוי\nמחפשת דף נחיתה לסדנאות שלי"
        t2 = "דוד כהן\nמחפש מתכנת לאפליקציית מובייל"
        assert _content_dedup_hash(t1) != _content_dedup_hash(t2)


    # ── emoji engagement lines ────────────────────────────────

    def test_emoji_reaction_line_removed(self):
        """שורות ריאקציה עם אמוג'י (👍 5) מוסרות — משתנות בין סריקות."""
        t1 = "פוסט על עבודה\n👍 5"
        t2 = "פוסט על עבודה\n👍❤️ 15"
        t3 = "פוסט על עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_emoji_only_line_removed(self):
        """שורות שהן רק אמוג'י ומספרים (ללא אותיות) מוסרות."""
        t1 = "פוסט על עבודה\n😂🔥\n👍 12"
        t2 = "פוסט על עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_emoji_engagement_with_words_removed(self):
        """'👍 5 תגובות' — engagement עם אמוג'י מוסר."""
        t1 = "פוסט על עבודה\n👍 5 תגובות"
        t2 = "פוסט על עבודה\n👍❤️ 12 תגובות · 3 שיתופים"
        t3 = "פוסט על עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_emoji_in_post_content_preserved(self):
        """אמוג'י בתוך תוכן פוסט (עם אותיות) נשמר."""
        text = "🏠 מחפש דירה 3 חדרים בתל אביב"
        result = _stable_text_for_hash(text)
        assert "מחפש דירה" in result

    # ── action bar ────────────────────────────────────────────

    def test_action_bar_combined_removed(self):
        """בר פעולות משולב ('אהבתי · תגובה · שיתוף') מוסר."""
        t1 = "תוכן הפוסט\nאהבתי · תגובה · שיתוף"
        t2 = "תוכן הפוסט"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_action_bar_english_removed(self):
        """English action bar ('Like · Comment · Share') removed."""
        t1 = "post content\nLike · Comment · Share"
        t2 = "post content"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    # ── Hebrew dates ──────────────────────────────────────────

    def test_hebrew_date_removed(self):
        """תאריכים עבריים ('4 במרץ') מוסרים — פייסבוק מחליף זמן יחסי בתאריך."""
        t1 = "ניב הראל\n8 שעות\nמחפש שותף טכנולוגי"
        t2 = "ניב הראל\n4 במרץ\nמחפש שותף טכנולוגי"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_hebrew_date_with_time_removed(self):
        """'4 במרץ בשעה 10:30' מוסר."""
        t1 = "ניב הראל\n4 במרץ בשעה 10:30\nמחפש שותף"
        t2 = "ניב הראל\nמחפש שותף"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_english_date_removed(self):
        """English dates ('March 4', 'Jan 15 at 10:30 AM') removed."""
        t1 = "Niv Harel\nMarch 4\nLooking for a tech partner"
        t2 = "Niv Harel\n8 hours\nLooking for a tech partner"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    def test_english_date_with_time_removed(self):
        """'Jan 15 at 10:30 AM' removed."""
        t1 = "Author\nJan 15 at 10:30 AM\npost content here"
        t2 = "Author\npost content here"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t2)

    # ── name-based reaction patterns ──────────────────────────

    def test_name_and_others_hebrew_removed(self):
        """'אמיר ו-3 אחרים' — רשימת מגיבים מוסרת."""
        t1 = "פוסט על עבודה\nאמיר ו-3 אחרים"
        t2 = "פוסט על עבודה\nדני, שרה ו-8 אחרים"
        t3 = "פוסט על עבודה"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    def test_name_and_others_english_removed(self):
        """'John and 3 others' — English reactors list removed."""
        t1 = "post content\nJohn and 3 others"
        t2 = "post content\nJohn, Jane and 5 others"
        t3 = "post content"
        assert _stable_text_for_hash(t1) == _stable_text_for_hash(t3)
        assert _stable_text_for_hash(t2) == _stable_text_for_hash(t3)

    # ── cross-session duplicate scenario (issue #115) ─────────

    def test_cross_session_same_post_different_timestamp_format(self):
        """תרחיש issue #115: אותו פוסט, סשן ראשון עם זמן יחסי, סשן שני עם תאריך."""
        from main import _content_dedup_hash
        # סשן 1: פוסט חדש עם זמן יחסי
        t1 = ("ניב הראל\n7 שעות\n"
              "מחפש שותף טכנולוגי לפרויקט SaaS שמבוסס על AI.\n"
              "אנחנו כבר בשלב MVP ומחפשים מישהו שיוביל את הפיתוח.\n"
              "👍 3\n2 תגובות\nאהבתי · תגובה · שיתוף")
        # סשן 2: אותו פוסט, 12 שעות אחרי, תאריך במקום זמן יחסי, יותר ריאקציות
        t2 = ("ניב הראל\n4 במרץ\n"
              "מחפש שותף טכנולוגי לפרויקט SaaS שמבוסס על AI.\n"
              "אנחנו כבר בשלב MVP ומחפשים מישהו שיוביל את הפיתוח.\n"
              "👍❤️ 15\n8 תגובות · 2 שיתופים\nאהבתי · תגובה · שיתוף\n"
              "דני כהן\nנשמע מעולה, שלח לי הודעה\nאהבתי\nהגב")
        h1 = _content_dedup_hash(t1)
        h2 = _content_dedup_hash(t2)
        assert h1 == h2, (
            f"content_hash חייב להיות זהה בין סשנים לאותו פוסט: "
            f"{h1} != {h2}"
        )


class TestContentHashDedup:
    """בדיקות ל-content_hash dedup — מניעת כפילויות כשה-post_id משתנה בין סשנים."""

    def test_is_content_hash_sent_empty(self, tmp_db):
        """hash ריק לא מסומן כנשלח."""
        assert database.is_content_hash_sent("") is False

    def test_is_content_hash_sent_not_found(self, tmp_db):
        """hash שלא קיים ב-DB מחזיר False."""
        assert database.is_content_hash_sent("abc123") is False

    def test_save_and_check_content_hash(self, tmp_db):
        """שמירת ליד עם content_hash — ואז בדיקה שהוא נמצא."""
        database.save_lead("post1", "group1", "content", "reason",
                           content_hash="hash_abc")
        assert database.is_content_hash_sent("hash_abc") is True

    def test_content_hash_catches_duplicate_with_different_id(self, tmp_db):
        """אותו content_hash עם post_id שונה — נתפס כנשלח."""
        database.save_lead("post1", "group1", "content", "reason",
                           content_hash="hash_abc")
        # post_id שונה (כמו שקורה כשה-hash משתנה בין סשנים)
        assert database.is_lead_sent("post2") is False  # ID שונה — לא נמצא
        assert database.is_content_hash_sent("hash_abc") is True  # תוכן זהה — נתפס

    def test_save_lead_without_content_hash(self, tmp_db):
        """save_lead ללא content_hash — backward compatible."""
        database.save_lead("post1", "group1", "content", "reason")
        assert database.is_lead_sent("post1") is True
        assert database.is_content_hash_sent("") is False


# ──────────────────────────────────────────────────────────────
# 6. classify_post (mocked API)
# ──────────────────────────────────────────────────────────────


@patch("classifier._CLASSIFICATION_CRITERIA_DEFAULT", "פרומפט סיווג לטסטים")
class TestClassifyPost:
    def _make_mock_response(self, text: str):
        """Build a mock OpenAI ChatCompletion response."""
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    @patch("classifier.client")
    def test_relevant_post(self, mock_client):
        from classifier import classify_post

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '{"relevant": true, "reason": "עסק קטן מחפש מפתח"}'
        )
        result = classify_post("מחפש מפתח לבוט", "קבוצת פרילנסרים")
        assert result["relevant"] is True
        assert "reason" in result

    @patch("classifier.client")
    def test_not_relevant_post(self, mock_client):
        from classifier import classify_post

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '{"relevant": false, "reason": "שיתוף מאמר"}'
        )
        result = classify_post("מאמר מעניין על AI", "טכנולוגיה")
        assert result["relevant"] is False

    @patch("classifier.client")
    def test_markdown_wrapped_json(self, mock_client):
        from classifier import classify_post

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '```json\n{"relevant": true, "reason": "ליד"}\n```'
        )
        result = classify_post("צריך מתכנת", "קבוצה")
        assert result["relevant"] is True

    @patch("classifier.client")
    def test_empty_response(self, mock_client):
        from classifier import classify_post

        mock_client.chat.completions.create.return_value = self._make_mock_response("")
        result = classify_post("פוסט", "קבוצה")
        assert result["relevant"] is False
        assert "ריקה" in result["reason"]

    @patch("classifier.client")
    def test_invalid_json(self, mock_client):
        from classifier import classify_post

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "this is not json"
        )
        result = classify_post("פוסט", "קבוצה")
        assert result["relevant"] is False

    @patch("classifier.client")
    def test_api_error(self, mock_client):
        import openai
        from classifier import classify_post

        mock_client.chat.completions.create.side_effect = openai.APIStatusError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "rate limit"}},
        )
        result = classify_post("פוסט", "קבוצה")
        assert result["relevant"] is False
        assert "API" in result["reason"]


# ──────────────────────────────────────────────────────────────
# 6b. classify_batch (mocked API)
# ──────────────────────────────────────────────────────────────


@patch("classifier._CLASSIFICATION_CRITERIA_DEFAULT", "פרומפט סיווג לטסטים")
class TestClassifyBatch:
    def _make_mock_response(self, text: str):
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    @patch("classifier.client")
    def test_batch_two_posts(self, mock_client):
        from classifier import classify_batch

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '[{"relevant": true, "reason": "מחפש מפתח"}, {"relevant": false, "reason": "שיתוף מאמר"}]'
        )
        posts = [
            {"content": "מחפש מפתח לבוט", "group": "קבוצה1"},
            {"content": "מאמר מעניין", "group": "קבוצה2"},
        ]
        results = classify_batch(posts, batch_size=5)
        assert len(results) == 2
        assert results[0]["relevant"] is True
        assert results[1]["relevant"] is False
        # נשלחה בקשה אחת בלבד (לא 2)
        assert mock_client.chat.completions.create.call_count == 1

    @patch("classifier.client")
    def test_batch_empty_list(self, mock_client):
        from classifier import classify_batch

        results = classify_batch([])
        assert results == []
        mock_client.chat.completions.create.assert_not_called()

    @patch("classifier.client")
    def test_batch_splits_into_batches(self, mock_client):
        from classifier import classify_batch

        def side_effect(**kwargs):
            msgs = kwargs.get("messages", [])
            # הודעת user היא האחרונה ברשימת messages
            user_msg = msgs[-1].get("content", "") if msgs else ""
            count = user_msg.count("--- פוסט")
            items = [{"relevant": False, "reason": "לא רלוונטי"}] * count
            return self._make_mock_response(json.dumps(items))

        mock_client.chat.completions.create.side_effect = side_effect
        posts = [{"content": f"פוסט {i}", "group": "קבוצה"} for i in range(7)]
        results = classify_batch(posts, batch_size=3)
        assert len(results) == 7
        # 7 פוסטים בבאצ'ים של 3: 3 + 3 + 1 = 3 קריאות API
        assert mock_client.chat.completions.create.call_count == 3

    @patch("classifier.client")
    def test_batch_single_post_returns_dict(self, mock_client):
        """באצ' של פוסט בודד — API יכול להחזיר אובייקט במקום מערך."""
        from classifier import classify_batch

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '{"relevant": true, "reason": "ליד"}'
        )
        results = classify_batch([{"content": "צריך מתכנת", "group": "קבוצה"}])
        assert len(results) == 1
        assert results[0]["relevant"] is True

    @patch("classifier.client")
    def test_batch_markdown_wrapped(self, mock_client):
        from classifier import classify_batch

        mock_client.chat.completions.create.return_value = self._make_mock_response(
            '```json\n[{"relevant": true, "reason": "ליד"}]\n```'
        )
        results = classify_batch([{"content": "צריך בוט", "group": "קבוצה"}])
        assert len(results) == 1
        assert results[0]["relevant"] is True

    @patch("classifier.client")
    def test_batch_fallback_on_length_mismatch(self, mock_client):
        """אם API מחזיר מערך בגודל שונה — fallback לסיווג בודד."""
        from classifier import classify_batch

        # קריאה ראשונה (batch) — מחזיר מערך בגודל שגוי
        batch_response = self._make_mock_response(
            '[{"relevant": true, "reason": "ליד"}]'  # מערך של 1 במקום 2
        )
        # קריאות fallback בודדות
        single_response = self._make_mock_response(
            '{"relevant": false, "reason": "לא רלוונטי"}'
        )
        mock_client.chat.completions.create.side_effect = [
            batch_response, single_response, single_response
        ]
        posts = [
            {"content": "פוסט 1", "group": "קבוצה"},
            {"content": "פוסט 2", "group": "קבוצה"},
        ]
        results = classify_batch(posts, batch_size=5)
        assert len(results) == 2
        # 1 batch + 2 fallback singles = 3 calls
        assert mock_client.chat.completions.create.call_count == 3

    @patch("classifier.client")
    def test_batch_fallback_on_api_error(self, mock_client):
        """אם API נופל בבאצ' — fallback לסיווג בודד פוסט-פוסט."""
        import openai
        from classifier import classify_batch

        single_ok = self._make_mock_response(
            '{"relevant": false, "reason": "fallback"}'
        )
        mock_client.chat.completions.create.side_effect = [
            openai.APIStatusError(
                message="server error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "server error"}},
            ),
            single_ok,
            single_ok,
        ]
        posts = [
            {"content": "פוסט 1", "group": "קבוצה"},
            {"content": "פוסט 2", "group": "קבוצה"},
        ]
        results = classify_batch(posts, batch_size=5)
        assert len(results) == 2
        assert all(r["relevant"] is False for r in results)

    @patch("classifier.client")
    def test_batch_fallback_on_null_element(self, mock_client):
        """אם אלמנט במערך הוא null — fallback בודד ללא תוצאות חלקיות."""
        from classifier import classify_batch

        # batch מחזיר מערך שבו אלמנט הוא None (לא dict)
        batch_response = self._make_mock_response(
            '[{"relevant": true, "reason": "ליד"}, null]'
        )
        single_ok = self._make_mock_response(
            '{"relevant": false, "reason": "fallback"}'
        )
        mock_client.chat.completions.create.side_effect = [
            batch_response, single_ok, single_ok
        ]
        posts = [
            {"content": "פוסט 1", "group": "קבוצה"},
            {"content": "פוסט 2", "group": "קבוצה"},
        ]
        results = classify_batch(posts, batch_size=5)
        # חייבים לקבל בדיוק 2 תוצאות — לא יותר
        assert len(results) == 2


# ──────────────────────────────────────────────────────────────
# 6c. Auto-model-fallback
# ──────────────────────────────────────────────────────────────


@patch("classifier._CLASSIFICATION_CRITERIA_DEFAULT", "פרומפט סיווג לטסטים")
class TestModelFallback:
    def _make_mock_response(self, text: str):
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    def setup_method(self):
        """איפוס מודל פעיל לפני כל טסט."""
        import classifier
        classifier._active_model = None

    @patch("classifier.client")
    def test_deprecated_model_rotates_to_next(self, mock_client):
        """מודל deprecated → עובר למודל הבא ברשימה ומצליח."""
        import openai
        import classifier

        classifier._active_model = "gpt-4.1-mini"
        ok_response = self._make_mock_response(
            '{"relevant": true, "reason": "ליד"}'
        )
        mock_client.chat.completions.create.side_effect = [
            openai.NotFoundError(
                message="model not found",
                response=MagicMock(status_code=404, headers={}),
                body={"error": {"message": "model not found"}},
            ),
            ok_response,
        ]
        result = classifier.classify_post("צריך מפתח", "קבוצה")
        assert result["relevant"] is True
        assert classifier._active_model == "gpt-4o-mini"

    @patch("classifier.client")
    def test_all_models_exhausted(self, mock_client):
        """כל המודלים deprecated → מחזיר שגיאה."""
        import openai
        import classifier

        classifier._active_model = "gpt-4o-mini"  # אחרון ברשימה
        mock_client.chat.completions.create.side_effect = openai.NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404, headers={}),
            body={"error": {"message": "model not found"}},
        )
        result = classifier.classify_post("פוסט", "קבוצה")
        assert result["relevant"] is False
        assert "API" in result["reason"]

    def test_rotate_model_returns_next(self):
        import classifier

        next_m = classifier._rotate_model("gpt-4.1-mini")
        assert next_m == "gpt-4o-mini"

    def test_rotate_model_last_wraps_around(self):
        """כשהמודל האחרון ברשימה נכשל, עוטף מסביב לראשון."""
        import classifier

        result = classifier._rotate_model("gpt-4o-mini")
        assert result == classifier._MODEL_PRIORITY[0]

    def test_is_model_deprecated_error_404(self):
        import openai
        import classifier

        err = openai.NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404, headers={}),
            body={},
        )
        assert classifier._is_model_deprecated_error(err) is True

    def test_non_deprecated_error_not_detected(self):
        import openai
        import classifier

        err = openai.APIStatusError(
            message="rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        assert classifier._is_model_deprecated_error(err) is False


# ──────────────────────────────────────────────────────────────
# 7. Database — Groups CRUD
# ──────────────────────────────────────────────────────────────

import sqlite3
import tempfile
from pathlib import Path

import database


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """יוצר DB זמני לבדיקות."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    # ניקוי חיבור thread-local קיים כדי שייפתח מחדש עם הנתיב החדש
    if hasattr(database._local, "conn"):
        del database._local.conn
    database.init_db()
    yield db_path
    if hasattr(database._local, "conn"):
        del database._local.conn


class TestGroupsCRUD:
    def test_add_group(self, tmp_db):
        ok, msg = database.add_group("https://www.facebook.com/groups/12345")
        assert ok is True
        assert "נוספה" in msg

    def test_add_group_duplicate(self, tmp_db):
        database.add_group("https://m.facebook.com/groups/12345")
        ok, msg = database.add_group("https://m.facebook.com/groups/12345")
        assert ok is False
        assert "כבר קיימת" in msg

    def test_remove_group(self, tmp_db):
        database.add_group("https://m.facebook.com/groups/99999")
        ok, msg = database.remove_group("https://m.facebook.com/groups/99999")
        assert ok is True
        assert "הוסרה" in msg

    def test_remove_group_not_found(self, tmp_db):
        ok, msg = database.remove_group("https://m.facebook.com/groups/nonexistent")
        assert ok is False
        assert "לא נמצאה" in msg

    def test_get_db_groups(self, tmp_db):
        database.add_group("https://m.facebook.com/groups/111")
        database.add_group("https://m.facebook.com/groups/222")
        groups = database.get_db_groups()
        assert len(groups) == 2
        assert groups[0]["name"] == "111"
        assert "m.facebook.com" in groups[0]["url"]

    def test_url_normalization(self, tmp_db):
        database.add_group("https://www.facebook.com/groups/55555/")
        groups = database.get_db_groups()
        assert "m.facebook.com" in groups[0]["url"]
        assert not groups[0]["url"].endswith("/")

    def test_url_normalization_schemeless(self, tmp_db):
        """URL בלי סכמה עם דומיין לא צריך לקבל כפילות prefix."""
        database.add_group("www.facebook.com/groups/77777")
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/77777"

    def test_url_normalization_bare_id(self, tmp_db):
        """מזהה קבוצה בלבד — מקבל prefix מלא."""
        database.add_group("12345")
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/12345"

    def test_url_normalization_bare_domain_with_scheme(self, tmp_db):
        """facebook.com (ללא www) עם סכמה צריך להיות m.facebook.com."""
        database.add_group("https://facebook.com/groups/88888")
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/88888"

    def test_url_normalization_bare_domain_without_scheme(self, tmp_db):
        """facebook.com (ללא www) בלי סכמה צריך להיות m.facebook.com."""
        database.add_group("facebook.com/groups/99999")
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/99999"

    def test_url_normalization_strips_query_params_and_fragment(self, tmp_db):
        """לינקי שיתוף עם ?ref=share&rdid=...# — צריך לנקות query ו-fragment."""
        database.add_group(
            "https://www.facebook.com/groups/189086514880040/"
            "?ref=share&rdid=qRJqlLa0GqkkzELG"
            "&share_url=https%3A%2F%2Fwww.facebook.com%2Fshare%2Fg%2F1QpBMPu2J2%2F#"
        )
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/189086514880040"
        assert groups[0]["name"] == "189086514880040"

    def test_never_configured_returns_none(self, tmp_db):
        assert database.get_db_groups() is None

    def test_add_then_remove_all_returns_empty_list(self, tmp_db):
        database.add_group("https://m.facebook.com/groups/111")
        database.remove_group("https://m.facebook.com/groups/111")
        groups = database.get_db_groups()
        assert groups == []


class TestGroupLimit:
    def test_no_limit_by_default(self, tmp_db, monkeypatch):
        """ללא MAX_GROUPS — אין הגבלה."""
        monkeypatch.delenv("MAX_GROUPS", raising=False)
        for i in range(20):
            ok, _ = database.add_group(f"https://m.facebook.com/groups/{i}")
            assert ok is True

    def test_limit_enforced(self, tmp_db, monkeypatch):
        """MAX_GROUPS=2 — לא ניתן להוסיף קבוצה שלישית."""
        monkeypatch.setenv("MAX_GROUPS", "2")
        ok1, _ = database.add_group("https://m.facebook.com/groups/aaa")
        ok2, _ = database.add_group("https://m.facebook.com/groups/bbb")
        ok3, msg3 = database.add_group("https://m.facebook.com/groups/ccc")
        assert ok1 is True
        assert ok2 is True
        assert ok3 is False
        assert "מגבלת" in msg3

    def test_limit_after_remove(self, tmp_db, monkeypatch):
        """אחרי מחיקת קבוצה — אפשר שוב להוסיף."""
        monkeypatch.setenv("MAX_GROUPS", "1")
        database.add_group("https://m.facebook.com/groups/first")
        ok, _ = database.add_group("https://m.facebook.com/groups/second")
        assert ok is False
        database.remove_group("https://m.facebook.com/groups/first")
        ok, _ = database.add_group("https://m.facebook.com/groups/second")
        assert ok is True

    def test_count_groups(self, tmp_db):
        assert database.count_groups() == 0
        database.add_group("https://m.facebook.com/groups/111")
        assert database.count_groups() == 1
        database.add_group("https://m.facebook.com/groups/222")
        assert database.count_groups() == 2

    def test_zero_means_unlimited(self, tmp_db, monkeypatch):
        """MAX_GROUPS=0 — ללא הגבלה."""
        monkeypatch.setenv("MAX_GROUPS", "0")
        for i in range(15):
            ok, _ = database.add_group(f"https://m.facebook.com/groups/{i}")
            assert ok is True


# ──────────────────────────────────────────────────────────────
# 8. Database — Keywords CRUD
# ──────────────────────────────────────────────────────────────


class TestKeywordsCRUD:
    def test_add_pre_filter_keyword(self, tmp_db):
        ok, msg = database.add_keyword("חדש", "pre_filter")
        assert ok is True
        assert "מילת מפתח" in msg

    def test_add_block_keyword(self, tmp_db):
        ok, msg = database.add_keyword("ספאם", "block")
        assert ok is True
        assert "מילה חסומה" in msg

    def test_add_duplicate_keyword(self, tmp_db):
        database.add_keyword("בוט", "pre_filter")
        ok, msg = database.add_keyword("בוט", "pre_filter")
        assert ok is False
        assert "כבר קיימת" in msg

    def test_same_word_different_types(self, tmp_db):
        ok1, _ = database.add_keyword("מילה", "pre_filter")
        ok2, _ = database.add_keyword("מילה", "block")
        assert ok1 is True
        assert ok2 is True

    def test_remove_keyword(self, tmp_db):
        database.add_keyword("למחוק", "block")
        ok, msg = database.remove_keyword("למחוק", "block")
        assert ok is True
        assert "הוסרה" in msg

    def test_remove_keyword_not_found(self, tmp_db):
        ok, msg = database.remove_keyword("אין", "block")
        assert ok is False
        assert "לא נמצאה" in msg

    def test_get_db_keywords(self, tmp_db):
        database.add_keyword("בוט", "pre_filter")
        database.add_keyword("סקריפט", "pre_filter")
        database.add_keyword("ספאם", "block")
        pf = database.get_db_keywords("pre_filter")
        bl = database.get_db_keywords("block")
        assert pf == ["בוט", "סקריפט"]
        assert bl == ["ספאם"]

    def test_empty_word_rejected(self, tmp_db):
        ok, msg = database.add_keyword("", "pre_filter")
        assert ok is False
        assert "ריקה" in msg

    def test_keyword_lowercased(self, tmp_db):
        database.add_keyword("SPAM", "block")
        kws = database.get_db_keywords("block")
        assert kws == ["spam"]

    def test_never_configured_returns_none(self, tmp_db):
        assert database.get_db_keywords("pre_filter") is None
        assert database.get_db_keywords("block") is None

    def test_add_then_remove_all_returns_empty_list(self, tmp_db):
        database.add_keyword("בוט", "pre_filter")
        database.remove_keyword("בוט", "pre_filter")
        kws = database.get_db_keywords("pre_filter")
        assert kws == []

    def test_cross_type_does_not_affect_other_type(self, tmp_db):
        """הוספת block לא צריכה לגרום ל-pre_filter להחזיר [] במקום None."""
        database.add_keyword("ספאם", "block")
        # block הוגדר — אבל pre_filter מעולם לא, אז צריך להחזיר None
        assert database.get_db_keywords("pre_filter") is None
        assert database.get_db_keywords("block") == ["ספאם"]

    def test_cross_type_independent_empty(self, tmp_db):
        """רוקנים סוג אחד — הסוג השני לא מושפע."""
        database.add_keyword("בוט", "pre_filter")
        database.add_keyword("ספאם", "block")
        database.remove_keyword("בוט", "pre_filter")
        # pre_filter רוקן בכוונה → []
        assert database.get_db_keywords("pre_filter") == []
        # block עדיין פעיל
        assert database.get_db_keywords("block") == ["ספאם"]


# ──────────────────────────────────────────────────────────────
# 9. reload_keywords
# ──────────────────────────────────────────────────────────────


class TestReloadKeywords:
    def test_reload_from_db(self, tmp_db):
        from main import reload_keywords
        database.add_keyword("חדשה", "pre_filter")
        database.add_keyword("חסימה", "block")
        reload_keywords()
        from main import PRE_FILTER_KEYWORDS, BLOCK_KEYWORDS
        assert "חדשה" in PRE_FILTER_KEYWORDS
        assert "חסימה" in BLOCK_KEYWORDS

    def test_fallback_to_defaults_when_never_configured(self, tmp_db):
        from main import reload_keywords, _PRE_FILTER_KEYWORDS_DEFAULT
        reload_keywords()
        from main import PRE_FILTER_KEYWORDS
        assert PRE_FILTER_KEYWORDS == _PRE_FILTER_KEYWORDS_DEFAULT

    def test_empty_list_when_all_removed(self, tmp_db):
        """הוספה ומחיקה של כל המילות מפתח צריכה להחזיר רשימה ריקה, לא defaults."""
        from main import reload_keywords
        database.add_keyword("זמני", "pre_filter")
        database.remove_keyword("זמני", "pre_filter")
        reload_keywords()
        from main import PRE_FILTER_KEYWORDS
        assert PRE_FILTER_KEYWORDS == []

    def test_block_keyword_does_not_kill_pre_filter_defaults(self, tmp_db):
        """הוספת block בלבד לא צריכה לרוקן את pre_filter — fallback ל-defaults."""
        from main import reload_keywords, _PRE_FILTER_KEYWORDS_DEFAULT
        database.add_keyword("ספאם", "block")
        reload_keywords()
        from main import PRE_FILTER_KEYWORDS, BLOCK_KEYWORDS
        assert PRE_FILTER_KEYWORDS == _PRE_FILTER_KEYWORDS_DEFAULT
        assert BLOCK_KEYWORDS == ["ספאם"]


# ──────────────────────────────────────────────────────────────
# 10. Health server
# ──────────────────────────────────────────────────────────────

import urllib.request

from main import _HealthHandler, start_health_server


class TestHealthServer:
    def test_returns_200_ok(self):
        """שרת healthcheck מחזיר 200 עם גוף 'ok'."""
        server = start_health_server(port=0)  # פורט 0 = OS בוחר פורט פנוי
        try:
            port = server.server_address[1]
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            assert resp.status == 200
            assert resp.read() == b"ok"
        finally:
            server.shutdown()

    def test_any_path_returns_200(self):
        """כל נתיב מחזיר 200 (לא רק /)."""
        server = start_health_server(port=0)
        try:
            port = server.server_address[1]
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
            assert resp.status == 200
        finally:
            server.shutdown()


# ──────────────────────────────────────────────────────────────
# 11. Database — get_config / set_config
# ──────────────────────────────────────────────────────────────


class TestConfigCRUD:
    def test_get_config_default(self, tmp_db):
        assert database.get_config("nonexistent") is None

    def test_get_config_custom_default(self, tmp_db):
        assert database.get_config("nonexistent", "fallback") == "fallback"

    def test_set_and_get_config(self, tmp_db):
        database.set_config("fb_email", "test@example.com")
        assert database.get_config("fb_email") == "test@example.com"

    def test_set_config_overwrites(self, tmp_db):
        database.set_config("interval", "10")
        database.set_config("interval", "20")
        assert database.get_config("interval") == "20"

    def test_set_config_does_not_break_feature_flags(self, tmp_db):
        """set_config לא צריך לשבור את _feature_was_configured."""
        database.set_config("custom_key", "value")
        database.add_group("https://m.facebook.com/groups/123")
        assert database._feature_was_configured("groups") is True
        assert database.get_config("custom_key") == "value"


# ──────────────────────────────────────────────────────────────
# 12. Panel API
# ──────────────────────────────────────────────────────────────

from panel import create_app


@pytest.fixture
def panel_client(tmp_db):
    """Flask test client עם DB זמני."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestPanelAPI:
    def test_get_settings(self, panel_client):
        resp = panel_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fb_email" in data
        assert "classification_criteria" in data
        assert "interval_minutes" in data

    def test_put_settings(self, panel_client):
        resp = panel_client.put("/api/settings", json={
            "fb_email": "new@test.com",
        })
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        # ערכים נשמרו
        resp2 = panel_client.get("/api/settings")
        data = resp2.get_json()
        assert data["fb_email"] == "new@test.com"

    def test_put_quiet_hours(self, panel_client):
        resp = panel_client.put("/api/settings", json={"quiet_hours": "22:00-06:00"})
        assert resp.status_code == 200
        resp2 = panel_client.get("/api/settings")
        assert resp2.get_json()["quiet_hours"] == "22:00-06:00"

    def test_get_groups_no_config(self, panel_client):
        """כשאין קבוצות מוגדרות — מחזיר רשימה ריקה עם source: none."""
        resp = panel_client.get("/api/groups")
        data = resp.get_json()
        assert data["source"] == "none"
        assert data["groups"] == []

    def test_add_and_list_groups(self, panel_client):
        resp = panel_client.post("/api/groups", json={"url": "https://facebook.com/groups/111"})
        assert resp.get_json()["ok"] is True
        resp2 = panel_client.get("/api/groups")
        data = resp2.get_json()
        assert data["source"] == "custom"
        assert any("111" in g["url"] for g in data["groups"])

    def test_delete_group(self, panel_client):
        panel_client.post("/api/groups", json={"url": "https://m.facebook.com/groups/222"})
        resp = panel_client.delete("/api/groups", json={"url": "https://m.facebook.com/groups/222"})
        assert resp.get_json()["ok"] is True

    def test_add_empty_group_rejected(self, panel_client):
        resp = panel_client.post("/api/groups", json={"url": ""})
        assert resp.status_code == 400

    def test_get_keywords_default(self, panel_client):
        resp = panel_client.get("/api/keywords/pre_filter")
        data = resp.get_json()
        assert data["source"] == "default"
        # ברירת מחדל מגיעה מ-env var — ייתכן שריקה אם לא הוגדר
        assert isinstance(data["keywords"], list)

    def test_add_and_list_keywords(self, panel_client):
        resp = panel_client.post("/api/keywords/pre_filter", json={"word": "טסט"})
        assert resp.get_json()["ok"] is True
        resp2 = panel_client.get("/api/keywords/pre_filter")
        data = resp2.get_json()
        assert data["source"] == "custom"
        assert "טסט" in data["keywords"]

    def test_add_and_list_block_keywords(self, panel_client):
        """הוספת מילה חסומה דרך הפאנל — בדיקת זרימה מלאה."""
        resp = panel_client.post("/api/keywords/block", json={"word": "ספאם"})
        data = resp.get_json()
        assert data["ok"] is True
        assert resp.status_code == 200

        resp2 = panel_client.get("/api/keywords/block")
        data2 = resp2.get_json()
        assert data2["source"] == "custom"
        assert "ספאם" in data2["keywords"]

    def test_delete_keyword(self, panel_client):
        panel_client.post("/api/keywords/block", json={"word": "ספאם"})
        resp = panel_client.delete("/api/keywords/block", json={"word": "ספאם"})
        assert resp.get_json()["ok"] is True

    def test_invalid_keyword_type(self, panel_client):
        resp = panel_client.get("/api/keywords/invalid")
        assert resp.status_code == 400

    def test_add_empty_keyword_rejected(self, panel_client):
        resp = panel_client.post("/api/keywords/pre_filter", json={"word": ""})
        assert resp.status_code == 400

    def test_settings_include_default_criteria(self, panel_client):
        """API מחזיר גם את ברירת המחדל של קריטריוני הסיווג."""
        resp = panel_client.get("/api/settings")
        data = resp.get_json()
        assert "classification_criteria_default" in data
        # ברירת מחדל מגיעה מ-env var — ייתכן שריקה אם לא הוגדר
        assert isinstance(data["classification_criteria_default"], str)

    def test_auth_not_required_when_no_token(self, panel_client):
        resp = panel_client.post("/api/auth")
        data = resp.get_json()
        assert data["ok"] is True
        assert data["auth_required"] is False

    def test_scan_status_returns_default_idle(self, panel_client):
        """בדיקה שה-endpoint מחזיר סטטוס idle כשאין סריקה."""
        resp = panel_client.get("/api/scan-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is False
        assert data["phase"] == "idle"
        assert "daily_seen" in data
        assert "server_time" in data

    def test_scan_status_reflects_progress_updates(self, panel_client):
        """בדיקה ש-scan_progress מ-main.py משתקף ב-endpoint — אותו dict בדיוק."""
        from main import scan_progress
        # שומרים ערכים מקוריים לשחזור
        orig = dict(scan_progress)
        try:
            scan_progress.update({
                "active": True,
                "phase": "scraping",
                "phase_label": "סורק קבוצות",
                "current_group": "test_group",
                "total_groups": 3,
            })
            resp = panel_client.get("/api/scan-status")
            data = resp.get_json()
            assert data["active"] is True
            assert data["phase"] == "scraping"
            assert data["phase_label"] == "סורק קבוצות"
            assert data["current_group"] == "test_group"
            assert data["total_groups"] == 3
        finally:
            scan_progress.update(orig)

    def test_cleared_quiet_hours_not_refilled_from_env(self, tmp_db, monkeypatch):
        """ערך ריק ב-DB (המשתמש ביטל) לא צריך לחזור ל-env var."""
        monkeypatch.setenv("QUIET_HOURS", "02:00-07:00")
        database.set_config("quiet_hours", "")
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/api/settings")
            data = resp.get_json()
            assert data["quiet_hours"] == ""


class TestPanelAuth:
    def test_auth_required_blocks_without_token(self, tmp_db, monkeypatch):
        """כשיש PANEL_TOKEN, בקשות ללא טוקן נחסמות."""
        monkeypatch.setenv("PANEL_TOKEN", "secret123")
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/api/settings")
            assert resp.status_code == 401

    def test_auth_passes_with_correct_token(self, tmp_db, monkeypatch):
        monkeypatch.setenv("PANEL_TOKEN", "secret123")
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/api/settings",
                              headers={"Authorization": "Bearer secret123"})
            assert resp.status_code == 200

    def test_html_page_accessible_without_token(self, tmp_db, monkeypatch):
        """דף ה-HTML נגיש גם בלי טוקן."""
        monkeypatch.setenv("PANEL_TOKEN", "secret123")
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────
# 13. Classifier — dynamic criteria loading
# ──────────────────────────────────────────────────────────────


class TestClassifierDynamicCriteria:
    def test_default_criteria(self, tmp_db):
        from classifier import _get_classification_criteria, _CLASSIFICATION_CRITERIA_DEFAULT
        result = _get_classification_criteria()
        assert result == _CLASSIFICATION_CRITERIA_DEFAULT

    def test_custom_criteria_from_db(self, tmp_db):
        from classifier import _get_classification_criteria
        database.set_config("classification_criteria", "קריטריונים מותאמים אישית")
        result = _get_classification_criteria()
        assert result == "קריטריונים מותאמים אישית"

    def test_empty_criteria_falls_back_to_default(self, tmp_db):
        from classifier import _get_classification_criteria, _CLASSIFICATION_CRITERIA_DEFAULT
        database.set_config("classification_criteria", "  ")
        result = _get_classification_criteria()
        assert result == _CLASSIFICATION_CRITERIA_DEFAULT

    def test_build_system_prompt_includes_json_instruction(self, tmp_db):
        from classifier import _build_system_prompt
        prompt = _build_system_prompt()
        assert "JSON" in prompt
        assert "relevant" in prompt

    def test_build_batch_prompt_includes_array_instruction(self, tmp_db):
        from classifier import _build_batch_system_prompt
        prompt = _build_batch_system_prompt()
        assert "מערך JSON" in prompt


# ──────────────────────────────────────────────────────────────
# 14. Main — dynamic settings loading
# ──────────────────────────────────────────────────────────────

from main import _load_interval_from_db, _load_quiet_hours_from_db


class TestDynamicSettings:
    def test_interval_from_env(self, tmp_db, monkeypatch):
        """interval נקרא ממשתנה סביבה בלבד, לא מ-DB."""
        monkeypatch.setattr("main.INTERVAL_MINUTES", 25)
        assert _load_interval_from_db() == 25

    def test_interval_zero_floored_to_one(self, tmp_db, monkeypatch):
        """ערך 0 מוחזר כ-1 — defense-in-depth מול לולאה צמודה."""
        monkeypatch.setattr("main.INTERVAL_MINUTES", 0)
        assert _load_interval_from_db() == 1

    def test_interval_ignores_stale_db_value(self, tmp_db, monkeypatch):
        """ערך ישן ב-DB לא דורס את env var."""
        database.set_config("interval_minutes", "99")
        monkeypatch.setattr("main.INTERVAL_MINUTES", 10)
        assert _load_interval_from_db() == 10

    def test_quiet_hours_from_db(self, tmp_db):
        database.set_config("quiet_hours", "22:00-06:00")
        result = _load_quiet_hours_from_db()
        assert result is not None
        assert result[0].hour == 22
        assert result[1].hour == 6

    def test_quiet_hours_empty_returns_none(self, tmp_db):
        database.set_config("quiet_hours", "")
        result = _load_quiet_hours_from_db()
        assert result is None

    def test_quiet_hours_invalid_returns_none(self, tmp_db):
        database.set_config("quiet_hours", "invalid")
        result = _load_quiet_hours_from_db()
        assert result is None

    def test_quiet_hours_empty_db_overrides_env(self, tmp_db, monkeypatch):
        """ערך ריק ב-DB צריך לבטל שעות שקטות גם אם QUIET_HOURS מוגדר ב-env."""
        monkeypatch.setattr("main.QUIET_HOURS", "02:00-07:00")
        database.set_config("quiet_hours", "")
        result = _load_quiet_hours_from_db()
        assert result is None


# ──────────────────────────────────────────────────────────────
# 15. Inline Keyboard — כפתורי תפריט
# ──────────────────────────────────────────────────────────────

from main import (
    _main_menu_buttons, _settings_menu_buttons, _back_to_menu_button,
    _build_status_text, _build_daily_report_text,
    _build_groups_text, _build_keywords_text, _build_blocked_text,
)


class TestMenuButtons:
    def test_main_menu_structure(self):
        """תפריט ראשי — 2 שורות, 2 כפתורים בכל שורה."""
        buttons = _main_menu_buttons()
        assert len(buttons) == 2
        assert len(buttons[0]) == 2
        assert len(buttons[1]) == 2

    def test_main_menu_callback_data(self):
        """כל כפתור בתפריט ראשי מכיל callback_data תקין."""
        buttons = _main_menu_buttons()
        expected = {"scan", "status", "daily_report", "settings"}
        actual = {btn["callback_data"] for row in buttons for btn in row}
        assert actual == expected

    def test_main_menu_buttons_have_text(self):
        """כל כפתור מכיל שדה text."""
        buttons = _main_menu_buttons()
        for row in buttons:
            for btn in row:
                assert "text" in btn
                assert len(btn["text"]) > 0

    def test_settings_menu_structure(self):
        """תפריט הגדרות — 3 שורות."""
        buttons = _settings_menu_buttons()
        assert len(buttons) == 3
        assert len(buttons[0]) == 2
        assert len(buttons[1]) == 2
        assert len(buttons[2]) == 1

    def test_settings_menu_callback_data(self):
        buttons = _settings_menu_buttons()
        expected = {"groups", "keywords", "blocked", "blocked_users", "menu"}
        actual = {btn["callback_data"] for row in buttons for btn in row}
        assert actual == expected

    def test_back_to_menu_button(self):
        """כפתור חזרה — שורה אחת, כפתור אחד."""
        buttons = _back_to_menu_button()
        assert len(buttons) == 1
        assert len(buttons[0]) == 1
        assert buttons[0][0]["callback_data"] == "menu"


class TestBuildStatusText:
    def test_contains_key_fields(self, tmp_db):
        shared = {
            "quiet": None,
            "scan_in_progress": False,
            "last_scan_started": None,
            "last_scan_finished": None,
        }
        text = _build_status_text(shared)
        assert "סטטוס בוט" in text
        assert "שעה מקומית" in text
        assert "סך נראו" in text

    def test_scan_in_progress(self, tmp_db):
        shared = {
            "quiet": None,
            "scan_in_progress": True,
            "last_scan_started": datetime(2026, 1, 1, 10, 0),
            "last_scan_finished": None,
        }
        text = _build_status_text(shared)
        assert "כן" in text


class TestBuildDailyReport:
    def test_contains_report_fields(self, tmp_db):
        shared = {
            "scan_in_progress": False,
            "last_scan_started": None,
            "last_scan_finished": None,
        }
        text = _build_daily_report_text(shared)
        assert "דו\"ח יומי" in text
        assert "פוסטים שנסרקו היום" in text
        assert "לידים שנשלחו היום" in text
        assert "סה\"כ כללי" in text


class TestBuildGroupsText:
    def test_no_groups(self):
        with patch("scraper.GROUPS", []):
            text = _build_groups_text()
            assert "אין קבוצות" in text

    def test_with_groups(self):
        fake = [{"name": "טסט", "url": "https://m.facebook.com/groups/123"}]
        with patch("scraper.GROUPS", fake):
            text = _build_groups_text()
            assert "טסט" in text
            assert "1." in text


class TestBuildKeywordsText:
    def test_empty_keywords(self):
        with patch.dict("main._keywords_state", {"pre_filter": [], "block": []}):
            text = _build_keywords_text()
            assert "(ריק)" in text

    def test_with_keywords(self):
        with patch.dict("main._keywords_state", {"pre_filter": ["בוט", "אוטומציה"], "block": ["ספאם"]}):
            text = _build_keywords_text()
            assert "בוט" in text
            assert "אוטומציה" in text
            assert "ספאם" in text
            assert "מילים חוסמות" in text


class TestBuildBlockedText:
    def test_empty_blocked(self):
        with patch.dict("main._keywords_state", {"block": []}):
            from main import _build_blocked_text
            text = _build_blocked_text()
            assert "(ריק)" in text

    def test_with_blocked(self):
        with patch.dict("main._keywords_state", {"block": ["ספאם"]}):
            from main import _build_blocked_text
            text = _build_blocked_text()
            assert "ספאם" in text


# ──────────────────────────────────────────────────────────────
# 16. Notifier — inline keyboard functions
# ──────────────────────────────────────────────────────────────

from notifier import send_message_with_buttons, edit_message_text, answer_callback_query


class TestSendMessageWithButtons:
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_sends_with_reply_markup(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_post.return_value = mock_resp

        buttons = [[{"text": "כפתור", "callback_data": "test"}]]
        result = send_message_with_buttons("טקסט", buttons, chat_id=123)
        assert result is not None
        assert result["message_id"] == 42

        call_payload = mock_post.call_args[1]["json"]
        assert "reply_markup" in call_payload
        assert call_payload["reply_markup"]["inline_keyboard"] == buttons

    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_returns_none_on_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        result = send_message_with_buttons("טקסט", [[]], chat_id=123)
        assert result is None

    def test_no_token_returns_none(self, monkeypatch):
        monkeypatch.setattr("notifier._get_bot_token", lambda: None)
        result = send_message_with_buttons("טקסט", [[]], chat_id=123)
        assert result is None


class TestEditMessageText:
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_edits_message(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        buttons = [[{"text": "כפתור", "callback_data": "test"}]]
        result = edit_message_text(123, 42, "טקסט חדש", buttons)
        assert result is True

        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["message_id"] == 42
        assert call_payload["text"] == "טקסט חדש"
        assert "reply_markup" in call_payload

    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_edit_without_buttons(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        result = edit_message_text(123, 42, "טקסט")
        assert result is True

        call_payload = mock_post.call_args[1]["json"]
        assert "reply_markup" not in call_payload

    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_returns_false_on_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.return_value = mock_resp

        result = edit_message_text(123, 42, "טקסט")
        assert result is False


class TestAnswerCallbackQuery:
    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_answers_query(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        result = answer_callback_query("cb_123", "הודעה")
        assert result is True

        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["callback_query_id"] == "cb_123"
        assert call_payload["text"] == "הודעה"

    @patch("notifier._get_bot_token", new=lambda: "fake-token")
    @patch("notifier.requests.post")
    def test_answers_without_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        answer_callback_query("cb_123")
        call_payload = mock_post.call_args[1]["json"]
        assert "text" not in call_payload

    def test_no_token_returns_false(self, monkeypatch):
        monkeypatch.setattr("notifier._get_bot_token", lambda: None)
        result = answer_callback_query("cb_123")
        assert result is False


# ──────────────────────────────────────────────────────────────
# 17. Panel — scan frequency removed
# ──────────────────────────────────────────────────────────────


class TestGetDailyStats:
    def test_daily_stats_filters_by_date(self, tmp_db):
        conn = database._get_conn()
        conn.execute("INSERT INTO seen_posts (post_id, group_name, seen_at) VALUES (?, ?, ?)",
                     ("p1", "g", "2026-02-26T10:00:00"))
        conn.execute("INSERT INTO seen_posts (post_id, group_name, seen_at) VALUES (?, ?, ?)",
                     ("p2", "g", "2026-02-25T10:00:00"))
        conn.execute("INSERT INTO sent_leads (post_id, group_name, content, reason, sent_at) VALUES (?, ?, ?, ?, ?)",
                     ("p1", "g", "c", "r", "2026-02-26T10:00:00"))
        conn.commit()
        stats = database.get_daily_stats("2026-02-26")
        assert stats["seen"] == 1
        assert stats["sent"] == 1

    def test_daily_stats_empty_day(self, tmp_db):
        stats = database.get_daily_stats("2026-02-26")
        assert stats["seen"] == 0
        assert stats["sent"] == 0


class TestPanelIntervalRemoved:
    def test_put_interval_ignored(self, panel_client):
        """interval_minutes כבר לא ניתן לעריכה מהפאנל."""
        resp = panel_client.put("/api/settings", json={"interval_minutes": "99"})
        assert resp.status_code == 200
        # ערך לא נשמר
        resp2 = panel_client.get("/api/settings")
        data = resp2.get_json()
        assert data["interval_minutes"] != "99"


# ──────────────────────────────────────────────────────────────
# 18. ensure_keywords_migrated — מיגרציית ברירות מחדל
# ──────────────────────────────────────────────────────────────


class TestEnsureKeywordsMigrated:
    def test_migrates_defaults_on_first_add(self, tmp_db):
        """הוספת מילה ראשונה צריכה להעביר ברירות מחדל ל-DB."""
        defaults = ["בוט", "אוטומציה", "סקריפט"]
        database.ensure_keywords_migrated("pre_filter", defaults)
        kws = database.get_db_keywords("pre_filter")
        assert set(kws) == {"בוט", "אוטומציה", "סקריפט"}

    def test_no_duplicate_on_second_call(self, tmp_db):
        """קריאה חוזרת לא מכפילה את המילים."""
        defaults = ["בוט", "אוטומציה"]
        database.ensure_keywords_migrated("pre_filter", defaults)
        database.ensure_keywords_migrated("pre_filter", defaults)
        kws = database.get_db_keywords("pre_filter")
        assert kws == ["בוט", "אוטומציה"]

    def test_add_after_migration_appends(self, tmp_db):
        """הוספת מילה חדשה אחרי מיגרציה — מצטרפת לברירות מחדל."""
        defaults = ["בוט", "אוטומציה"]
        database.ensure_keywords_migrated("pre_filter", defaults)
        database.add_keyword("חדשה", "pre_filter")
        kws = database.get_db_keywords("pre_filter")
        assert "חדשה" in kws
        assert "בוט" in kws
        assert "אוטומציה" in kws

    def test_empty_defaults_still_marks_configured(self, tmp_db):
        """ברירות מחדל ריקות — לא מוסיפות כלום אבל לא שוברות."""
        database.ensure_keywords_migrated("block", [])
        # רשימה ריקה לא מסמנת כמוגדר — _feature_was_configured נשאר False
        assert database.get_db_keywords("block") is None

    def test_does_not_overwrite_existing(self, tmp_db):
        """אם הסוג כבר הוגדר — לא דורסים."""
        database.add_keyword("קיימת", "pre_filter")
        database.ensure_keywords_migrated("pre_filter", ["חדשה1", "חדשה2"])
        kws = database.get_db_keywords("pre_filter")
        # רק המילה שהוספנו ידנית — ברירות מחדל לא נוספו
        assert kws == ["קיימת"]

    def test_panel_add_preserves_defaults(self, panel_client):
        """הוספת מילה בפאנל שומרת את ברירות המחדל מ-env."""
        # שלב 1: הגדרת ברירות מחדל דרך env (כמו שהלקוח יגדיר)
        test_defaults = ["בוט", "אוטומציה", "סקריפט"]
        with patch("main._PRE_FILTER_KEYWORDS_DEFAULT", test_defaults):
            # שלב 2: לפני הוספה — ברירות מחדל
            resp = panel_client.get("/api/keywords/pre_filter")
            data = resp.get_json()
            assert data["source"] == "default"
            default_count = len(data["keywords"])
            assert default_count == 3

            # שלב 3: הוספת מילה חדשה דרך הפאנל
            resp = panel_client.post("/api/keywords/pre_filter", json={"word": "מילה_חדשה_מאוד"})
            assert resp.get_json()["ok"] is True

            # שלב 4: בדיקה שכל ברירות המחדל + החדשה קיימות
            resp = panel_client.get("/api/keywords/pre_filter")
            data = resp.get_json()
            assert data["source"] == "custom"
            assert "מילה_חדשה_מאוד" in data["keywords"]
            # ברירות המחדל נשמרו + המילה החדשה
            assert len(data["keywords"]) == default_count + 1


    def test_panel_add_without_existing(self, panel_client):
        """הוספת קבוצה ראשונה — ללא ברירות מחדל."""
        # שלב 1: לפני הוספה — אין קבוצות
        resp = panel_client.get("/api/groups")
        data = resp.get_json()
        assert data["source"] == "none"
        assert data["groups"] == []

        # שלב 2: הוספת קבוצה חדשה דרך הפאנל
        resp = panel_client.post("/api/groups", json={"url": "https://m.facebook.com/groups/test_new_group_99999"})
        assert resp.get_json()["ok"] is True

        # שלב 3: בדיקה שרק הקבוצה החדשה קיימת
        resp = panel_client.get("/api/groups")
        data = resp.get_json()
        assert data["source"] == "custom"
        urls = [g["url"] for g in data["groups"]]
        assert "https://m.facebook.com/groups/test_new_group_99999" in urls
        assert len(data["groups"]) == 1


# ──────────────────────────────────────────────────────────────
# 19. update_group_name — עדכון שם קבוצה ב-DB
# ──────────────────────────────────────────────────────────────


class TestUpdateGroupName:
    def test_update_existing_group(self, tmp_db):
        """עדכון שם קבוצה קיימת ב-DB."""
        database.add_group("https://m.facebook.com/groups/12345")
        groups = database.get_db_groups()
        assert groups[0]["name"] == "12345"

        updated = database.update_group_name("https://m.facebook.com/groups/12345", "שם אמיתי")
        assert updated is True

        groups = database.get_db_groups()
        assert groups[0]["name"] == "שם אמיתי"

    def test_update_nonexistent_group(self, tmp_db):
        """עדכון קבוצה שלא קיימת מחזיר False."""
        updated = database.update_group_name("https://m.facebook.com/groups/99999", "שם")
        assert updated is False

    def test_update_preserves_url(self, tmp_db):
        """עדכון שם לא משנה את ה-URL."""
        database.add_group("https://m.facebook.com/groups/55555")
        database.update_group_name("https://m.facebook.com/groups/55555", "קבוצה חדשה")
        groups = database.get_db_groups()
        assert groups[0]["url"] == "https://m.facebook.com/groups/55555"
        assert groups[0]["name"] == "קבוצה חדשה"


# ──────────────────────────────────────────────────────────────
# 20. _to_desktop_url — המרת URL מובייל לדסקטופ
# ──────────────────────────────────────────────────────────────

from notifier import _to_desktop_url


class TestToDesktopUrl:
    def test_mobile_to_desktop(self):
        url = "https://m.facebook.com/groups/123/posts/456"
        assert _to_desktop_url(url) == "https://www.facebook.com/groups/123/posts/456"

    def test_already_desktop(self):
        url = "https://www.facebook.com/groups/123/posts/456"
        assert _to_desktop_url(url) == url

    def test_no_facebook(self):
        url = "https://example.com/test"
        assert _to_desktop_url(url) == url

    def test_group_url_converted(self):
        url = "https://m.facebook.com/groups/1684554685829832"
        assert _to_desktop_url(url) == "https://www.facebook.com/groups/1684554685829832"


# ──────────────────────────────────────────────────────────────
# 21. _is_post_url — זיהוי URL של פוסט ספציפי
# ──────────────────────────────────────────────────────────────

from scraper import _is_post_url


class TestIsPostUrl:
    def test_posts_pattern(self):
        assert _is_post_url("https://m.facebook.com/groups/123/posts/456") is True

    def test_permalink_pattern(self):
        assert _is_post_url("https://m.facebook.com/groups/123/permalink/456") is True

    def test_story_fbid_pattern(self):
        assert _is_post_url("https://m.facebook.com/story.php?story_fbid=123") is True

    def test_pfbid_pattern(self):
        assert _is_post_url("https://m.facebook.com/groups/123/posts/pfbidABC") is True

    def test_group_url_only(self):
        """URL של קבוצה בלבד — לא פוסט ספציפי."""
        assert _is_post_url("https://m.facebook.com/groups/123") is False

    def test_p_pattern(self):
        assert _is_post_url("https://m.facebook.com/p/something") is True

    def test_multi_permalinks(self):
        assert _is_post_url("https://m.facebook.com/groups/123/?multi_permalinks=456") is True

    def test_share_p_pattern(self):
        """פורמט share link חדש של פייסבוק."""
        assert _is_post_url("https://www.facebook.com/share/p/1AkvNbuYNi/") is True

    def test_share_p_mobile(self):
        """share link מאתר מובייל."""
        assert _is_post_url("https://m.facebook.com/share/p/Xb2cD3fG/") is True


# ──────────────────────────────────────────────────────────────
# 24. _extract_name_from_title — חילוץ שם קבוצה מכותרת דף
# ──────────────────────────────────────────────────────────────

from scraper import _extract_name_from_title


class TestExtractNameFromTitle:
    def test_standard_title(self):
        assert _extract_name_from_title("בינה מלאכותית AI ISRAEL | Facebook") == "בינה מלאכותית AI ISRAEL"

    def test_no_separator(self):
        """כותרת ללא ' | ' — לא ניתן לחלץ."""
        assert _extract_name_from_title("Facebook") is None

    def test_empty_string(self):
        assert _extract_name_from_title("") is None

    def test_none_input(self):
        assert _extract_name_from_title(None) is None

    def test_bad_title_login(self):
        """כותרת כניסה — לא שם קבוצה אמיתי."""
        assert _extract_name_from_title("Log in | Facebook") is None

    def test_bad_title_hebrew(self):
        assert _extract_name_from_title("התחברות | Facebook") is None

    def test_bad_title_error(self):
        assert _extract_name_from_title("Error | Facebook") is None

    def test_short_name_rejected(self):
        """שם קצר מדי (2 תווים או פחות) נדחה."""
        assert _extract_name_from_title("AB | Facebook") is None

    def test_unsupported_browser(self):
        """כותרת 'אינו נתמך' נדחית."""
        assert _extract_name_from_title("הדפדפן אינו נתמך | Facebook") is None

    def test_html_entities_unescaped(self):
        """HTML entities מפוענחים כראוי."""
        assert _extract_name_from_title("AI &amp; ML Israel | Facebook") == "AI & ML Israel"

    def test_multiple_separators(self):
        """כותרת עם כמה מפרידים — רק החלק הראשון."""
        assert _extract_name_from_title("שם קבוצה | תת-כותרת | Facebook") == "שם קבוצה"

    def test_no_separator_valid_name(self):
        """כותרת ללא ' | ' אבל עם שם תקין — מחזיר את השם."""
        assert _extract_name_from_title("בניית אתרים, בוני אתרים בישראל.") == "בניית אתרים, בוני אתרים בישראל."

    def test_lrm_marks_stripped(self):
        """תווי כיווניות (LRM) מנוקים מהכותרת."""
        assert _extract_name_from_title("\u200eשם קבוצה\u200e") == "שם קבוצה"

    def test_rlm_marks_stripped(self):
        """תווי כיווניות (RLM) מנוקים מהכותרת."""
        assert _extract_name_from_title("\u200fשם קבוצה\u200f") == "שם קבוצה"


# ──────────────────────────────────────────────────────────────
# 25. _extract_id_from_data_attrs — חילוץ מזהה פוסט מ-data attributes
# ──────────────────────────────────────────────────────────────

from scraper import _extract_id_from_data_attrs


class TestExtractIdFromDataAttrs:
    def test_mf_story_key(self):
        """מחלץ mf_story_key מ-JSON."""
        data = '{"mf_story_key": "123456789"}'
        assert _extract_id_from_data_attrs(data) == "123456789"

    def test_top_level_post_id(self):
        """מחלץ top_level_post_id כשאין mf_story_key."""
        data = '{"top_level_post_id": "987654321"}'
        assert _extract_id_from_data_attrs(data) == "987654321"

    def test_tl_objid(self):
        """מחלץ tl_objid כשאין מזהים אחרים."""
        data = '{"tl_objid": "555666777"}'
        assert _extract_id_from_data_attrs(data) == "555666777"

    def test_priority_order(self):
        """mf_story_key קודם ל-top_level_post_id."""
        data = '{"top_level_post_id": "111", "mf_story_key": "222"}'
        assert _extract_id_from_data_attrs(data) == "222"

    def test_invalid_json(self):
        """JSON לא תקין — מחזיר None."""
        assert _extract_id_from_data_attrs("not-json") is None

    def test_empty_json(self):
        """JSON ריק — מחזיר None."""
        assert _extract_id_from_data_attrs("{}") is None

    def test_none_input(self):
        """None — מחזיר None."""
        assert _extract_id_from_data_attrs(None) is None

    def test_empty_string_value(self):
        """ערך ריק — מחזיר None."""
        data = '{"mf_story_key": ""}'
        assert _extract_id_from_data_attrs(data) is None

    def test_numeric_value(self):
        """ערך מספרי (לא מחרוזת) — מומר למחרוזת."""
        data = '{"mf_story_key": 123456789}'
        assert _extract_id_from_data_attrs(data) == "123456789"


# ──────────────────────────────────────────────────────────────
# 26. _extract_post_url — אסטרטגיות חילוץ URL (async, עם מוקים)
# ──────────────────────────────────────────────────────────────

from scraper import _extract_post_url, get_extraction_stats, reset_extraction_stats


class _FakeLink:
    """לינק מדומה עם href ו-text."""

    def __init__(self, href, text=""):
        self._href = href
        self._text = text

    async def get_attribute(self, name):
        if name == "href":
            return self._href
        return None

    async def inner_text(self):
        return self._text


class _FakeElementForStrategies:
    """אלמנט מדומה גמיש יותר — תומך בכל האסטרטגיות."""

    def __init__(self, *, selector_links=None, all_links=None, attrs=None,
                 inner_els=None, el_id=None, html="<div></div>",
                 climb_result=None, text=""):
        # selector_links: dict of CSS selector -> href
        self._selector_links = selector_links or {}
        # all_links: list of (href, text) — לסריקה כללית
        self._all_links = all_links or []
        self._attrs = attrs or {}
        self._inner_els = inner_els or {}
        self._el_id = el_id
        self._html = html
        # climb_result: תוצאת JS climbing (אסטרטגיה 6) — JSON string או None
        self._climb_result = climb_result
        self._text = text

    async def query_selector(self, selector):
        if selector in self._inner_els:
            return self._inner_els[selector]
        if selector in self._selector_links:
            return _FakeLink(self._selector_links[selector])
        return None

    async def query_selector_all(self, selector):
        if "a[href]" in selector:
            return [_FakeLink(href, text) for href, text in self._all_links]
        return []

    async def get_attribute(self, name):
        if name == "id":
            return self._el_id
        return self._attrs.get(name)

    async def evaluate(self, expression, arg=None):
        """מדמה el.evaluate() — מחזיר climb_result אם הוגדר (אסטרטגיה 6)."""
        return self._climb_result

    async def inner_html(self):
        return self._html

    async def inner_text(self):
        return self._text


class TestExtractPostUrl:
    """טסטים ל-_extract_post_url — כל 7 האסטרטגיות."""

    FALLBACK = "https://m.facebook.com/groups/123/"

    @pytest.mark.asyncio
    async def test_strategy1_direct_selector(self):
        """אסטרטגיה 1 — סלקטור ישיר לקישור פוסט."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            selector_links={"a[href*='/posts/']": "/groups/123/posts/456"}
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/posts/456"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy1_share_link(self):
        """אסטרטגיה 1 — סלקטור ישיר ל-share link חדש."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            selector_links={"a[href*='/share/p/']": "https://www.facebook.com/share/p/1AkvNbuYNi/"}
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://www.facebook.com/share/p/1AkvNbuYNi/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy1_full_url(self):
        """אסטרטגיה 1 — URL מלא (עם http) לא מקבל prefix."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            selector_links={"a[href*='/posts/']": "https://m.facebook.com/groups/123/posts/456"}
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/posts/456"

    @pytest.mark.asyncio
    async def test_strategy2_link_scan(self):
        """אסטרטגיה 2 — סריקת כל הלינקים."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            all_links=[
                ("https://m.facebook.com/groups/123", ""),
                ("/groups/123/permalink/789", ""),
            ]
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/789"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy2_share_link_scan(self):
        """אסטרטגיה 2 — סריקת לינקים מוצאת share link."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            all_links=[
                ("https://m.facebook.com/groups/123", ""),
                ("https://www.facebook.com/share/p/1AkvNbuYNi/", ""),
            ]
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://www.facebook.com/share/p/1AkvNbuYNi/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy3_timestamp_hebrew(self):
        """אסטרטגיה 3 — לינק timestamp בעברית."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            all_links=[
                ("/groups/123/permalink/999888", "3 שעות"),
            ]
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert "999888" in result
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy3_timestamp_english(self):
        """אסטרטגיה 3 — לינק timestamp באנגלית."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            all_links=[
                ("/groups/123/permalink/111222", "5h"),
            ]
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert "111222" in result

    @pytest.mark.asyncio
    async def test_strategy4_data_ft(self):
        """אסטרטגיה 4 — חילוץ מזהה מ-data-ft."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            attrs={"data-ft": '{"mf_story_key": "555666"}'}
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/555666/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy4_data_store(self):
        """אסטרטגיה 4 — חילוץ מזהה מ-data-store."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            attrs={"data-store": '{"top_level_post_id": "777888"}'}
        )
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/777888/"

    @pytest.mark.asyncio
    async def test_strategy4_inner_data_ft(self):
        """אסטרטגיה 4 — data-ft באלמנט פנימי."""
        reset_extraction_stats()
        inner = _FakeElementForStrategies(
            attrs={"data-ft": '{"mf_story_key": "999111"}'}
        )
        el = _FakeElementForStrategies(
            inner_els={"[data-ft]": inner}
        )
        # צריך שהאלמנט החיצוני לא יחזיר data-ft בעצמו
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/999111/"

    @pytest.mark.asyncio
    async def test_strategy5_element_id(self):
        """אסטרטגיה 5 — מזהה מספרי ב-id של האלמנט."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(el_id="mall_post_1234567890")
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/1234567890/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_fallback_when_all_fail(self):
        """כשכל האסטרטגיות נכשלות — מחזיר fallback URL."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(html="<div>no links</div>")
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == self.FALLBACK
        assert get_extraction_stats()["fallback"] == 1

    @pytest.mark.asyncio
    async def test_fallback_counter_increments(self):
        """מונה fallback עולה בכל כישלון."""
        reset_extraction_stats()
        el = _FakeElementForStrategies()
        await _extract_post_url(el, self.FALLBACK)
        await _extract_post_url(el, self.FALLBACK)
        assert get_extraction_stats()["fallback"] == 2

    @pytest.mark.asyncio
    async def test_strategy4_no_group_id_in_fallback(self):
        """אסטרטגיה 4 לא עובדת אם ה-fallback URL לא מכיל group ID."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(
            attrs={"data-ft": '{"mf_story_key": "555666"}'}
        )
        result = await _extract_post_url(el, "https://m.facebook.com/")
        # אין group ID ב-fallback, אז אסטרטגיה 4 לא מופעלת — fallback
        assert result == "https://m.facebook.com/"

    @pytest.mark.asyncio
    async def test_strategy5_short_id_ignored(self):
        """אסטרטגיה 5 — מזהה קצר מ-10 ספרות לא נחשב."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(el_id="post_12345")
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == self.FALLBACK

    @pytest.mark.asyncio
    async def test_strategy6_dom_climbing_post_link(self):
        """אסטרטגיה 6 — טיפוס ב-DOM מוצא קישור פוסט."""
        reset_extraction_stats()
        import json
        climb = json.dumps({"url": "https://m.facebook.com/groups/123/posts/777/", "lvl": 3, "how": "post_link"})
        el = _FakeElementForStrategies(climb_result=climb)
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/posts/777/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy6_dom_climbing_timestamp(self):
        """אסטרטגיה 6 — טיפוס ב-DOM מוצא לינק timestamp."""
        reset_extraction_stats()
        import json
        climb = json.dumps({"url": "https://m.facebook.com/groups/123/permalink/888/", "lvl": 7, "how": "timestamp"})
        el = _FakeElementForStrategies(climb_result=climb)
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/888/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy6_dom_climbing_ajaxify(self):
        """אסטרטגיה 6 — טיפוס ב-DOM מוצא ajaxify עם URL פוסט."""
        reset_extraction_stats()
        import json
        climb = json.dumps({"url": "https://m.facebook.com/groups/123/permalink/999/", "lvl": 5, "how": "ajaxify"})
        el = _FakeElementForStrategies(climb_result=climb)
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == "https://m.facebook.com/groups/123/permalink/999/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy6_no_result_falls_through(self):
        """אסטרטגיה 6 — אם JS climbing לא מוצא כלום, ממשיכים ל-fallback."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(climb_result=None)
        result = await _extract_post_url(el, self.FALLBACK)
        assert result == self.FALLBACK
        assert get_extraction_stats()["fallback"] == 1

    @pytest.mark.asyncio
    async def test_strategy7_page_level_match(self):
        """אסטרטגיה 7 — התאמה לקישור מסריקת דף לפי תוכן טקסט."""
        reset_extraction_stats()
        post_text = "אני מחפש מפתח Fullstack עם ניסיון בפרויקטים גדולים ומורכבים"
        page_urls = [
            {"url": "https://m.facebook.com/share/p/ABC123/",
             "text": f"דוד כהן\n3 שעות\n{post_text}\n5 תגובות"},
        ]
        el = _FakeElementForStrategies(climb_result=None, text=post_text)
        result = await _extract_post_url(
            el, self.FALLBACK, page_urls=page_urls
        )
        assert result == "https://m.facebook.com/share/p/ABC123/"
        assert get_extraction_stats()["success"] == 1

    @pytest.mark.asyncio
    async def test_strategy7_no_match_falls_to_fallback(self):
        """אסטרטגיה 7 — אם טקסט האלמנט לא נמצא בסריקת הדף, fallback."""
        reset_extraction_stats()
        page_urls = [
            {"url": "https://m.facebook.com/share/p/ABC123/",
             "text": "טקסט שונה לגמרי מהאלמנט"},
        ]
        el = _FakeElementForStrategies(
            climb_result=None, text="תוכן ייחודי שלא מופיע בשום מקום אחר בדף"
        )
        result = await _extract_post_url(
            el, self.FALLBACK, page_urls=page_urls
        )
        assert result == self.FALLBACK
        assert get_extraction_stats()["fallback"] == 1

    @pytest.mark.asyncio
    async def test_strategy7_short_text_skipped(self):
        """אסטרטגיה 7 — טקסט קצר מ-30 תווים לא מנסה התאמה (למנוע false positive)."""
        reset_extraction_stats()
        page_urls = [
            {"url": "https://m.facebook.com/share/p/ABC123/",
             "text": "כותרת קצרה"},
        ]
        el = _FakeElementForStrategies(
            climb_result=None, text="כותרת קצרה"
        )
        result = await _extract_post_url(
            el, self.FALLBACK, page_urls=page_urls
        )
        assert result == self.FALLBACK

    @pytest.mark.asyncio
    async def test_strategy7_empty_page_urls(self):
        """אסטרטגיה 7 — רשימה ריקה לא משנה התנהגות."""
        reset_extraction_stats()
        el = _FakeElementForStrategies(climb_result=None, text="טקסט ארוך מספיק לבדיקה")
        result = await _extract_post_url(
            el, self.FALLBACK, page_urls=[]
        )
        assert result == self.FALLBACK


# ──────────────────────────────────────────────────────────────
# 27. extraction stats — סטטיסטיקת חילוץ URL
# ──────────────────────────────────────────────────────────────


class TestExtractionStats:
    def test_reset(self):
        """reset_extraction_stats מאפס את המונים."""
        reset_extraction_stats()
        stats = get_extraction_stats()
        assert stats["success"] == 0
        assert stats["fallback"] == 0

    def test_returns_copy(self):
        """get_extraction_stats מחזיר עותק — שינוי לא משפיע על המקור."""
        reset_extraction_stats()
        stats = get_extraction_stats()
        stats["success"] = 999
        assert get_extraction_stats()["success"] == 0


# ──────────────────────────────────────────────────────────────
# extract_post_age_days — חילוץ גיל פוסט מתוך טקסט
# ──────────────────────────────────────────────────────────────

from main import extract_post_age_days, is_post_too_old


class TestExtractPostAgeDays:
    """בדיקות לחילוץ גיל פוסט מתוך הטקסט."""

    # ── עברית — "לפני X [יחידה]" ──
    def test_hebrew_seconds(self):
        age = extract_post_age_days("לפני 10 שניות\nטקסט של פוסט")
        assert age is not None
        assert age < 0.001  # 10 שניות = כמעט 0 ימים

    def test_hebrew_minutes(self):
        age = extract_post_age_days("לפני 5 דקות\nטקסט של פוסט")
        assert age is not None
        assert age < 0.01  # 5 דקות = פחות מ-0.01 יום

    def test_hebrew_hours(self):
        age = extract_post_age_days("לפני 3 שעות\nמחפש מפתח פייתון")
        assert age is not None
        assert 0.1 < age < 0.2  # 3/24 ≈ 0.125

    def test_hebrew_days(self):
        age = extract_post_age_days("לפני 2 ימים\nדרוש מתכנת")
        assert age == 2.0

    def test_hebrew_days_singular(self):
        age = extract_post_age_days("לפני 1 יום")
        assert age == 1.0

    def test_hebrew_weeks(self):
        age = extract_post_age_days("לפני 2 שבועות\nפוסט ישן")
        assert age == 14.0

    def test_hebrew_months(self):
        age = extract_post_age_days("לפני 3 חודשים")
        assert age == 90.0

    # ── עברית — יחידות בודדות ──
    def test_hebrew_single_minute(self):
        age = extract_post_age_days("לפני דקה\nטקסט")
        assert age is not None
        assert age < 0.001

    def test_hebrew_single_hour(self):
        age = extract_post_age_days("לפני שעה\nטקסט")
        assert age is not None
        assert 0.03 < age < 0.05  # 1/24 ≈ 0.042

    def test_hebrew_single_day(self):
        age = extract_post_age_days("לפני יום\nטקסט")
        assert age == 1.0

    def test_hebrew_two_days(self):
        age = extract_post_age_days("לפני יומיים")
        assert age == 2.0

    def test_hebrew_single_week(self):
        age = extract_post_age_days("לפני שבוע")
        assert age == 7.0

    def test_hebrew_single_month(self):
        age = extract_post_age_days("לפני חודש")
        assert age == 30.0

    def test_hebrew_single_year(self):
        age = extract_post_age_days("לפני שנה")
        assert age == 365.0

    def test_hebrew_two_years(self):
        age = extract_post_age_days("לפני שנתיים")
        assert age == 730.0

    # ── מילים עצמאיות ──
    def test_now_hebrew(self):
        age = extract_post_age_days("עכשיו\nטקסט של פוסט")
        assert age == 0.0

    def test_yesterday_hebrew(self):
        age = extract_post_age_days("אתמול\nטקסט של פוסט")
        assert age == 1.0

    def test_just_now_english(self):
        age = extract_post_age_days("Just now\nSome post text")
        assert age == 0.0

    def test_yesterday_english(self):
        age = extract_post_age_days("Yesterday\nSome post")
        assert age == 1.0

    # ── פורמט קצר ──
    def test_short_hours(self):
        age = extract_post_age_days("3h\nטקסט")
        assert age is not None
        assert 0.1 < age < 0.2

    def test_short_days(self):
        age = extract_post_age_days("2d\nטקסט")
        assert age == 2.0

    def test_short_weeks(self):
        age = extract_post_age_days("1w\nטקסט")
        assert age == 7.0

    def test_short_minutes(self):
        age = extract_post_age_days("45m\nטקסט")
        assert age is not None
        assert age < 0.04

    # ── אנגלית ──
    def test_english_hours(self):
        age = extract_post_age_days("6 hours ago\nLooking for developer")
        assert age is not None
        assert age == 6 / 24.0

    def test_english_days(self):
        age = extract_post_age_days("3 days\nold post")
        assert age == 3.0

    def test_english_days_ago(self):
        age = extract_post_age_days("2 days ago\npost content")
        assert age == 2.0

    # ── false-positive prevention: ביטויי זמן בתוך גוף הטקסט ──
    def test_no_false_positive_day_trial(self):
        """'30 day trial' בתוך משפט — לא timestamp."""
        age = extract_post_age_days("Get our 30 day trial for free!\nSign up now")
        assert age is None

    def test_no_false_positive_month_project(self):
        """'3 month project' — לא timestamp."""
        age = extract_post_age_days("Looking for a developer for a 3 month project")
        assert age is None

    def test_no_false_positive_hour_session(self):
        """'2 hour session' — לא timestamp."""
        age = extract_post_age_days("Join our 2 hour session on Python")
        assert age is None

    def test_english_standalone_line(self):
        """שורה עצמאית '5 minutes' — כן timestamp."""
        age = extract_post_age_days("User Name\n5 minutes\nPost content here")
        assert age is not None
        assert age < 0.01

    # ── אין timestamp ──
    def test_no_timestamp(self):
        age = extract_post_age_days("מחפש מתכנת פייתון לפרויקט. מישהו מכיר?")
        assert age is None

    def test_empty_string(self):
        age = extract_post_age_days("")
        assert age is None

    # ── timestamp מוטמע בתוך טקסט ארוך ──
    def test_timestamp_embedded(self):
        text = "שם המשתמש\nלפני 5 שעות\nמחפש עזרה עם בוט טלגרם\n3 תגובות"
        age = extract_post_age_days(text)
        assert age is not None
        assert 0.2 < age < 0.22  # 5/24 ≈ 0.208

    def test_no_false_positive_hebrew_seconds_in_sentence(self):
        """ביטוי זמן בתוך משפט בעברית — לא timestamp."""
        age = extract_post_age_days("דוגמה: לפני 10 שניות קיבלתי הודעה וזה לא timestamp")
        assert age is None


class TestIsPostTooOld:
    """בדיקות לפונקציית is_post_too_old."""

    def test_old_post_filtered(self):
        """פוסט ישן מ-2 ימים עם סף 1 — מסונן."""
        assert is_post_too_old("לפני 3 ימים\nפוסט ישן", max_days=2) is True

    def test_young_post_passes(self):
        """פוסט צעיר מ-3 שעות עם סף 1 — עובר."""
        assert is_post_too_old("לפני 3 שעות\nפוסט חדש", max_days=1) is False

    def test_no_timestamp_passes(self):
        """פוסט ללא timestamp — עובר (שמרני, לא מסננים)."""
        assert is_post_too_old("מחפש מתכנת", max_days=1) is False

    def test_exact_threshold(self):
        """פוסט בדיוק בסף — מסונן (>= max_days)."""
        assert is_post_too_old("לפני 2 ימים\nטקסט", max_days=2) is True

    def test_just_under_threshold(self):
        """פוסט מתחת לסף — עובר."""
        assert is_post_too_old("לפני 1 ימים\nטקסט", max_days=2) is False

    def test_hours_under_threshold(self):
        """פוסט של שעות עם סף 1 יום — עובר."""
        assert is_post_too_old("לפני 23 שעות\nטקסט", max_days=1) is False

    def test_weeks_old(self):
        """פוסט ישן של שבוע עם סף 3 ימים — מסונן."""
        assert is_post_too_old("לפני 1 שבועות\nטקסט", max_days=3) is True


# ──────────────────────────────────────────────────────────────
# מעקב עלויות API (issue #93)
# ──────────────────────────────────────────────────────────────

from database import (
    save_api_usage, get_usage_stats, get_daily_usage_stats,
    get_usage_by_model, _estimate_cost,
)


class TestEstimateCost:
    """בדיקות חישוב עלות לפי מודל וטוקנים."""

    def test_gpt4_1_mini_cost(self):
        """עלות gpt-4.1-mini: $0.40/1M input, $1.60/1M output."""
        cost = _estimate_cost("gpt-4.1-mini", 1_000_000, 1_000_000)
        assert abs(cost - 2.0) < 0.001

    def test_gpt4o_mini_cost(self):
        """עלות gpt-4o-mini: $0.15/1M input, $0.60/1M output."""
        cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 0.001

    def test_unknown_model_uses_default(self):
        """מודל לא מוכר — fallback לעלות ברירת מחדל."""
        cost = _estimate_cost("unknown-model", 1_000_000, 1_000_000)
        assert abs(cost - 2.0) < 0.001

    def test_zero_tokens(self):
        cost = _estimate_cost("gpt-4.1-mini", 0, 0)
        assert cost == 0.0

    def test_small_call(self):
        """קריאה טיפוסית: 500 input + 200 output tokens."""
        cost = _estimate_cost("gpt-4.1-mini", 500, 200)
        expected = (500 * 0.40 + 200 * 1.60) / 1_000_000
        assert abs(cost - expected) < 1e-9


class TestSaveApiUsage:
    """בדיקות שמירה ושליפת נתוני שימוש ב-API."""

    def test_save_and_get_stats(self, tmp_db):
        """שמירת שימוש ושליפה מצטברת."""
        save_api_usage("gpt-4.1-mini", 100, 50, 150, "single")
        save_api_usage("gpt-4.1-mini", 200, 100, 300, "batch")
        stats = get_usage_stats()
        assert stats["prompt_tokens"] == 300
        assert stats["completion_tokens"] == 150
        assert stats["total_tokens"] == 450
        assert stats["total_calls"] == 2
        assert stats["total_cost_usd"] > 0

    def test_empty_stats(self, tmp_db):
        """ללא נתונים — מחזיר אפסים."""
        stats = get_usage_stats()
        assert stats["total_tokens"] == 0
        assert stats["total_calls"] == 0
        assert stats["total_cost_usd"] == 0

    def test_daily_stats(self, tmp_db):
        """סטטיסטיקות יומיות מסננות לפי תאריך."""
        save_api_usage("gpt-4.1-mini", 100, 50, 150, "single")
        today = database._now().strftime("%Y-%m-%d")
        daily = get_daily_usage_stats(today)
        assert daily["total_calls"] == 1
        assert daily["total_tokens"] == 150
        # תאריך אחר — אין תוצאות
        other = get_daily_usage_stats("1999-01-01")
        assert other["total_calls"] == 0

    def test_by_model(self, tmp_db):
        """פירוט לפי מודל."""
        save_api_usage("gpt-4.1-mini", 100, 50, 150, "single")
        save_api_usage("gpt-4o-mini", 200, 100, 300, "batch")
        by_model = get_usage_by_model()
        assert len(by_model) == 2
        models = {m["model"] for m in by_model}
        assert "gpt-4.1-mini" in models
        assert "gpt-4o-mini" in models


class TestTrackUsage:
    """בדיקות _track_usage ב-classifier.py."""

    def test_track_usage_saves_to_db(self, tmp_db):
        """_track_usage שומר נתונים ל-DB כשיש usage בתשובה."""
        from classifier import _track_usage

        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class FakeResponse:
            usage = FakeUsage()

        _track_usage(FakeResponse(), "gpt-4.1-mini", "single")
        stats = get_usage_stats()
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 150

    def test_track_usage_no_usage_attr(self, tmp_db):
        """_track_usage לא קורס כשאין usage בתשובה."""
        from classifier import _track_usage

        class FakeResponse:
            usage = None

        _track_usage(FakeResponse(), "gpt-4.1-mini", "single")
        stats = get_usage_stats()
        assert stats["total_calls"] == 0

    def test_track_usage_missing_attr(self, tmp_db):
        """_track_usage לא קורס כשאין שדה usage כלל."""
        from classifier import _track_usage

        class FakeResponse:
            pass

        _track_usage(FakeResponse(), "gpt-4.1-mini", "single")
        stats = get_usage_stats()
        assert stats["total_calls"] == 0


class TestBuildDeveloperUsageText:
    """בדיקות לפונקציית _build_developer_usage_text."""

    def test_empty_usage(self, tmp_db, monkeypatch):
        """ללא נתונים — מציג אפסים."""
        from main import _build_developer_usage_text
        monkeypatch.setattr("main.TIMEZONE_NAME", "UTC")
        text = _build_developer_usage_text()
        assert "קריאות: 0" in text
        assert "$0.0000" in text

    def test_with_data(self, tmp_db, monkeypatch):
        """עם נתונים — מציג סטטיסטיקות."""
        from main import _build_developer_usage_text
        monkeypatch.setattr("main.TIMEZONE_NAME", "UTC")
        save_api_usage("gpt-4.1-mini", 1000, 500, 1500, "single")
        text = _build_developer_usage_text()
        assert "1,500" in text  # total tokens formatted
        assert "gpt-4.1-mini" in text


class TestParseDeveloperChatIds:
    """בדיקות _parse_developer_chat_ids."""

    def test_empty(self, monkeypatch):
        monkeypatch.setattr("main.DEVELOPER_CHAT_ID", "")
        from main import _parse_developer_chat_ids
        assert _parse_developer_chat_ids() == set()

    def test_single_id(self, monkeypatch):
        monkeypatch.setattr("main.DEVELOPER_CHAT_ID", "12345")
        from main import _parse_developer_chat_ids
        assert _parse_developer_chat_ids() == {12345}

    def test_multiple_ids(self, monkeypatch):
        monkeypatch.setattr("main.DEVELOPER_CHAT_ID", "111,222")
        from main import _parse_developer_chat_ids
        assert _parse_developer_chat_ids() == {111, 222}


# ──────────────────────────────────────────────────────────────
# Group-Specific Force Send
# ──────────────────────────────────────────────────────────────

from main import (
    add_group_force_send_keyword,
    remove_group_force_send_keyword,
    _load_group_force_send_keywords,
    get_all_group_force_send,
)


class TestGroupForceSendCRUD:
    """הוספה/מחיקה/טעינה של מילות force_send לקבוצה ספציפית."""

    def test_add_and_load(self, tmp_db):
        ok, msg = add_group_force_send_keyword("https://m.facebook.com/groups/123", "skill")
        assert ok
        kws = _load_group_force_send_keywords("https://m.facebook.com/groups/123")
        assert "skill" in kws

    def test_add_duplicate(self, tmp_db):
        add_group_force_send_keyword("https://m.facebook.com/groups/123", "skill")
        ok, msg = add_group_force_send_keyword("https://m.facebook.com/groups/123", "skill")
        assert not ok

    def test_remove(self, tmp_db):
        add_group_force_send_keyword("https://m.facebook.com/groups/123", "skill")
        ok, msg = remove_group_force_send_keyword("https://m.facebook.com/groups/123", "skill")
        assert ok
        kws = _load_group_force_send_keywords("https://m.facebook.com/groups/123")
        assert "skill" not in kws

    def test_remove_not_found(self, tmp_db):
        ok, msg = remove_group_force_send_keyword("https://m.facebook.com/groups/123", "nonexistent")
        assert not ok

    def test_lowercased(self, tmp_db):
        add_group_force_send_keyword("https://m.facebook.com/groups/123", "SKILL")
        kws = _load_group_force_send_keywords("https://m.facebook.com/groups/123")
        assert "skill" in kws

    def test_url_normalization(self, tmp_db):
        """כתובות שונות לאותה קבוצה — מנורמלות לאותו מפתח."""
        add_group_force_send_keyword("www.facebook.com/groups/123", "skill")
        kws = _load_group_force_send_keywords("https://m.facebook.com/groups/123")
        assert "skill" in kws

    def test_different_groups_isolated(self, tmp_db):
        """מילות force_send בין קבוצות שונות מבודדות."""
        add_group_force_send_keyword("https://m.facebook.com/groups/aaa", "word_a")
        add_group_force_send_keyword("https://m.facebook.com/groups/bbb", "word_b")
        kws_a = _load_group_force_send_keywords("https://m.facebook.com/groups/aaa")
        kws_b = _load_group_force_send_keywords("https://m.facebook.com/groups/bbb")
        assert "word_a" in kws_a and "word_b" not in kws_a
        assert "word_b" in kws_b and "word_a" not in kws_b

    def test_get_all(self, tmp_db):
        """get_all_group_force_send מחזיר את כל הקבוצות שיש להן מילים."""
        add_group_force_send_keyword("https://m.facebook.com/groups/aaa", "w1")
        add_group_force_send_keyword("https://m.facebook.com/groups/bbb", "w2")
        all_gfs = get_all_group_force_send()
        assert len(all_gfs) == 2
        assert "w1" in all_gfs["https://m.facebook.com/groups/aaa"]
        assert "w2" in all_gfs["https://m.facebook.com/groups/bbb"]

    def test_empty_group_not_in_get_all(self, tmp_db):
        """קבוצה ללא מילים לא מופיעה ב-get_all."""
        add_group_force_send_keyword("https://m.facebook.com/groups/aaa", "w1")
        remove_group_force_send_keyword("https://m.facebook.com/groups/aaa", "w1")
        all_gfs = get_all_group_force_send()
        assert "https://m.facebook.com/groups/aaa" not in all_gfs


class TestMatchesForceSendWithGroup:
    """matches_force_send — בדיקת מילות גלובליות + קבוצה ספציפית."""

    def test_global_match(self):
        """מילה גלובלית תואמת ללא group_url."""
        with patch.dict("main._keywords_state", {"force_send": ["skill"], "group_force_send": {}}):
            assert matches_force_send("I have a Skill") == "skill"

    def test_group_match(self):
        """מילה ספציפית לקבוצה תואמת כשמועבר group_url מתאים."""
        with patch.dict("main._keywords_state", {"force_send": [], "group_force_send": {
                 "https://m.facebook.com/groups/123": ["getdrip"]
             }}):
            result = matches_force_send(
                "GetDRIP is great",
                group_url="https://m.facebook.com/groups/123",
            )
            assert result == "getdrip"

    def test_group_match_wrong_group(self):
        """מילה ספציפית לקבוצה לא תואמת כשה-group_url שונה."""
        with patch.dict("main._keywords_state", {"force_send": [], "group_force_send": {
                 "https://m.facebook.com/groups/123": ["getdrip"]
             }}):
            result = matches_force_send(
                "GetDRIP is great",
                group_url="https://m.facebook.com/groups/999",
            )
            assert result is None

    def test_global_before_group(self):
        """מילה גלובלית קודמת למילת קבוצה."""
        with patch.dict("main._keywords_state", {"force_send": ["global_word"], "group_force_send": {
                 "https://m.facebook.com/groups/123": ["group_word"]
             }}):
            result = matches_force_send(
                "text with global_word and group_word",
                group_url="https://m.facebook.com/groups/123",
            )
            assert result == "global_word"

    def test_no_match_anywhere(self):
        """אין התאמה לא גלובלית ולא לקבוצה."""
        with patch.dict("main._keywords_state", {"force_send": ["skill"], "group_force_send": {
                 "https://m.facebook.com/groups/123": ["getdrip"]
             }}):
            result = matches_force_send(
                "just a regular post",
                group_url="https://m.facebook.com/groups/123",
            )
            assert result is None

    def test_no_group_url_skips_group_check(self):
        """ללא group_url — בודק רק גלובליות."""
        with patch.dict("main._keywords_state", {"force_send": [], "group_force_send": {
                 "https://m.facebook.com/groups/123": ["getdrip"]
             }}):
            result = matches_force_send("GetDRIP is great")
            assert result is None


class TestGroupForceSendPanelAPI:
    """בדיקות API endpoints של group_force_send בפאנל."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_db):
        monkeypatch.setenv("PANEL_TOKEN", "")
        from panel import create_app
        app = create_app()
        self.client = app.test_client()

    def test_list_empty(self):
        resp = self.client.get("/api/group_force_send")
        data = resp.get_json()
        assert data["groups"] == {}

    def test_add_and_list(self):
        resp = self.client.post("/api/group_force_send/keywords",
                                json={"url": "https://m.facebook.com/groups/123", "word": "skill"})
        assert resp.get_json()["ok"]

        resp = self.client.get("/api/group_force_send/keywords?url=https://m.facebook.com/groups/123")
        data = resp.get_json()
        assert "skill" in data["keywords"]

    def test_add_and_list_all(self):
        self.client.post("/api/group_force_send/keywords",
                         json={"url": "https://m.facebook.com/groups/aaa", "word": "w1"})
        self.client.post("/api/group_force_send/keywords",
                         json={"url": "https://m.facebook.com/groups/bbb", "word": "w2"})
        resp = self.client.get("/api/group_force_send")
        data = resp.get_json()
        assert "w1" in data["groups"]["https://m.facebook.com/groups/aaa"]
        assert "w2" in data["groups"]["https://m.facebook.com/groups/bbb"]

    def test_delete(self):
        self.client.post("/api/group_force_send/keywords",
                         json={"url": "https://m.facebook.com/groups/123", "word": "skill"})
        resp = self.client.delete("/api/group_force_send/keywords",
                                  json={"url": "https://m.facebook.com/groups/123", "word": "skill"})
        assert resp.get_json()["ok"]

        resp = self.client.get("/api/group_force_send/keywords?url=https://m.facebook.com/groups/123")
        assert resp.get_json()["keywords"] == []

    def test_empty_url_rejected(self):
        resp = self.client.post("/api/group_force_send/keywords",
                                json={"url": "", "word": "skill"})
        assert resp.status_code == 400

    def test_empty_word_rejected(self):
        resp = self.client.post("/api/group_force_send/keywords",
                                json={"url": "https://m.facebook.com/groups/123", "word": ""})
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────
# get_config_by_prefix (database.py)
# ──────────────────────────────────────────────────────────────

from database import get_config_by_prefix, set_config


class TestGetConfigByPrefix:
    """בדיקת get_config_by_prefix — שמשמש את get_all_group_force_send."""

    def test_returns_matching_rows(self, tmp_db):
        set_config("force_send_group:aaa", '["w1"]')
        set_config("force_send_group:bbb", '["w2"]')
        set_config("other_key", "val")
        rows = get_config_by_prefix("force_send_group:")
        assert len(rows) == 2
        keys = [r[0] for r in rows]
        assert "force_send_group:aaa" in keys
        assert "force_send_group:bbb" in keys

    def test_no_matches(self, tmp_db):
        set_config("other_key", "val")
        rows = get_config_by_prefix("force_send_group:")
        assert rows == []

    def test_underscore_in_prefix_not_wildcard(self, tmp_db):
        """_ ב-prefix לא פועל כ-wildcard — רק התאמה מדויקת."""
        set_config("test_key", "exact")
        set_config("testXkey", "wrong")
        rows = get_config_by_prefix("test_key")
        keys = [r[0] for r in rows]
        assert "test_key" in keys
        assert "testXkey" not in keys


# ──────────────────────────────────────────────────────────────
# JSON Logging
# ──────────────────────────────────────────────────────────────

class TestJsonLogging:
    """בדיקת פורמט JSON logging."""

    def test_json_formatter_output(self):
        import logging
        from logger import _JsonFormatter
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert parsed["logger"] == "test"
        assert "time" in parsed

    def test_json_formatter_with_exception(self):
        import logging
        from logger import _JsonFormatter
        fmt = _JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="boom", args=(), exc_info=exc_info,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


# ──────────────────────────────────────────────────────────────
# Deep Health Check
# ──────────────────────────────────────────────────────────────

class TestDeepHealthCheck:
    """בדיקת _deep_health_check."""

    def test_healthy_with_recent_scan(self, tmp_db, monkeypatch):
        from datetime import timezone
        import main
        now = datetime.now(tz=timezone.utc)
        monkeypatch.setattr(main, "_health_shared_state", {
            "scan_in_progress": False,
            "last_scan_started": now,
            "last_scan_finished": now,
        })
        monkeypatch.setattr(main, "_HEALTH_MAX_SCAN_AGE_MINUTES", 60)
        # mock SESSION_FILE exists
        from unittest.mock import PropertyMock
        with patch("scraper.SESSION_FILE") as mock_sf:
            mock_sf.exists.return_value = True
            status, details = main._deep_health_check()
        assert status == 200
        assert details["status"] == "healthy"
        assert details["db"] == "ok"

    def test_unhealthy_missing_session(self, tmp_db, monkeypatch):
        from datetime import timezone
        import main
        now = datetime.now(tz=timezone.utc)
        monkeypatch.setattr(main, "_health_shared_state", {
            "scan_in_progress": False,
            "last_scan_started": now,
            "last_scan_finished": now,
        })
        monkeypatch.setattr(main, "_HEALTH_MAX_SCAN_AGE_MINUTES", 60)
        with patch("scraper.SESSION_FILE") as mock_sf:
            mock_sf.exists.return_value = False
            status, details = main._deep_health_check()
        assert status == 503
        assert details["session_file"] == "missing"

    def test_no_shared_state(self, tmp_db, monkeypatch):
        import main
        monkeypatch.setattr(main, "_health_shared_state", None)
        with patch("scraper.SESSION_FILE") as mock_sf:
            mock_sf.exists.return_value = True
            status, details = main._deep_health_check()
        assert details["last_scan"] == "not_initialized"

    def test_disk_error_marks_unhealthy(self, tmp_db, monkeypatch):
        """שגיאת disk_usage מסמנת unhealthy."""
        from datetime import timezone
        import shutil
        import main
        now = datetime.now(tz=timezone.utc)
        monkeypatch.setattr(main, "_health_shared_state", {
            "scan_in_progress": False,
            "last_scan_started": now,
            "last_scan_finished": now,
        })
        monkeypatch.setattr(main, "_HEALTH_MAX_SCAN_AGE_MINUTES", 60)
        monkeypatch.setattr(shutil, "disk_usage", lambda _: (_ for _ in ()).throw(OSError("disk err")))
        with patch("scraper.SESSION_FILE") as mock_sf:
            mock_sf.exists.return_value = True
            status, details = main._deep_health_check()
        assert status == 503
        assert "error" in details["disk"]


# ──────────────────────────────────────────────────────────────
# Blocked Users — CRUD + filtering
# ──────────────────────────────────────────────────────────────

import main
from main import is_user_blocked


class TestBlockedUsersCRUD:
    def test_add_blocked_user(self, tmp_db):
        ok, msg = database.add_blocked_user("https://www.facebook.com/spammer123")
        assert ok is True
        assert "נחסם" in msg

    def test_add_blocked_user_duplicate(self, tmp_db):
        database.add_blocked_user("https://m.facebook.com/spammer123")
        ok, msg = database.add_blocked_user("https://www.facebook.com/spammer123")
        assert ok is False
        assert "כבר חסום" in msg

    def test_add_blocked_user_empty(self, tmp_db):
        ok, msg = database.add_blocked_user("")
        assert ok is False
        assert "ריק" in msg

    def test_add_blocked_user_with_display_name(self, tmp_db):
        ok, msg = database.add_blocked_user(
            "https://m.facebook.com/spammer123", display_name="ספאמר מציק"
        )
        assert ok is True
        users = database.get_blocked_users()
        assert users[0]["name"] == "ספאמר מציק"
        assert "spammer123" in users[0]["profile_url"]

    def test_add_blocked_user_extracts_name_from_url(self, tmp_db):
        database.add_blocked_user("https://m.facebook.com/john.doe")
        users = database.get_blocked_users()
        assert users[0]["name"] == "john.doe"

    def test_remove_blocked_user(self, tmp_db):
        database.add_blocked_user("https://m.facebook.com/spammer123")
        ok, msg = database.remove_blocked_user("https://m.facebook.com/spammer123")
        assert ok is True
        assert "הוסר" in msg

    def test_remove_blocked_user_not_found(self, tmp_db):
        ok, msg = database.remove_blocked_user("https://m.facebook.com/nonexistent")
        assert ok is False
        assert "לא נמצא" in msg

    def test_get_blocked_users(self, tmp_db):
        database.add_blocked_user("https://m.facebook.com/user1", display_name="אלון")
        database.add_blocked_user("https://m.facebook.com/user2", display_name="דני")
        users = database.get_blocked_users()
        assert len(users) == 2
        names = [u["name"] for u in users]
        assert "אלון" in names
        assert "דני" in names

    def test_get_blocked_users_empty(self, tmp_db):
        users = database.get_blocked_users()
        assert users == []

    def test_url_normalization(self, tmp_db):
        """www → m, הסרת query params."""
        database.add_blocked_user("https://www.facebook.com/spammer?fbclid=abc")
        users = database.get_blocked_users()
        assert "m.facebook.com" in users[0]["profile_url"]
        assert "fbclid" not in users[0]["profile_url"]

    def test_profile_php_url_preserves_id(self, tmp_db):
        """profile.php?id=12345 — שומר את ה-id, מסיר tracking params."""
        database.add_blocked_user("https://www.facebook.com/profile.php?id=12345&fbclid=abc")
        users = database.get_blocked_users()
        assert "id=12345" in users[0]["profile_url"]
        assert "fbclid" not in users[0]["profile_url"]
        assert users[0]["name"] == "12345"

    def test_profile_php_two_different_ids(self, tmp_db):
        """שני משתמשי profile.php שונים לא מתנגשים."""
        ok1, _ = database.add_blocked_user("https://facebook.com/profile.php?id=111")
        ok2, _ = database.add_blocked_user("https://facebook.com/profile.php?id=222")
        assert ok1 is True
        assert ok2 is True
        assert database.get_blocked_users().__len__() == 2

    def test_same_name_different_urls(self, tmp_db):
        """שני משתמשים עם אותו שם אבל URLs שונים — שניהם נחסמים."""
        ok1, _ = database.add_blocked_user("https://facebook.com/user1", display_name="ספאמר")
        ok2, _ = database.add_blocked_user("https://facebook.com/user2", display_name="ספאמר")
        assert ok1 is True
        assert ok2 is True
        assert len(database.get_blocked_users()) == 2


class TestIsUserBlocked:
    _SPAMMER = {"name": "ספאמר", "profile_url": "https://m.facebook.com/spammer123"}
    _ANNOYING = {"name": "מציק", "profile_url": "https://m.facebook.com/annoying456"}

    def test_empty_list(self):
        with patch.dict("main._keywords_state", {"blocked_users": []}):
            assert is_user_blocked("https://m.facebook.com/anyone") is False

    def test_blocked_user(self):
        with patch.dict("main._keywords_state", {"blocked_users": [self._SPAMMER, self._ANNOYING]}):
            assert is_user_blocked("https://m.facebook.com/spammer123") is True

    def test_not_blocked(self):
        with patch.dict("main._keywords_state", {"blocked_users": [self._SPAMMER]}):
            assert is_user_blocked("https://m.facebook.com/normal_user") is False

    def test_www_normalized(self):
        """URL עם www צריך להתאים ל-m.facebook.com ב-DB."""
        with patch.dict("main._keywords_state", {"blocked_users": [self._SPAMMER]}):
            assert is_user_blocked("https://www.facebook.com/spammer123") is True

    def test_empty_author_url(self):
        with patch.dict("main._keywords_state", {"blocked_users": [self._SPAMMER]}):
            assert is_user_blocked("") is False

    def test_none_author_url(self):
        """author_url=None לא צריך לקרוס — מחזיר False."""
        with patch.dict("main._keywords_state", {"blocked_users": [self._SPAMMER]}):
            assert is_user_blocked(None) is False

    def test_exact_match_required(self):
        """חסימה לפי URL מלא, לא substring."""
        user = {"name": "אלון", "profile_url": "https://m.facebook.com/alon"}
        with patch.dict("main._keywords_state", {"blocked_users": [user]}):
            assert is_user_blocked("https://m.facebook.com/alon123") is False
            assert is_user_blocked("https://m.facebook.com/alon") is True


class TestBuildBlockedUsersText:
    def test_with_users(self):
        users = [
            {"name": "אלון", "profile_url": "https://m.facebook.com/alon"},
            {"name": "דני", "profile_url": "https://m.facebook.com/dani"},
        ]
        with patch.dict("main._keywords_state", {"blocked_users": users}):
            text = main._build_blocked_users_text()
            assert "אלון" in text
            assert "דני" in text
            assert "(2)" in text
            # URL מוצג כדי שהמשתמש יוכל לעשות /unblock_user
            assert "m.facebook.com/alon" in text
            assert "m.facebook.com/dani" in text

    def test_empty(self):
        with patch.dict("main._keywords_state", {"blocked_users": []}):
            text = main._build_blocked_users_text()
            assert "(ריק)" in text
            assert "(0)" in text


# ──────────────────────────────────────────────────────────────
# Cross-group dedup — פוסט זהה מקבוצות שונות נשלח פעם אחת
# ──────────────────────────────────────────────────────────────

class TestCrossGroupDedup:
    """בדיקות לאיחוד פוסטים זהים מקבוצות שונות בתוך סשן סריקה."""

    def test_send_lead_also_in_display(self):
        """send_lead מציג '+ N קבוצות נוספות' כשיש also_in."""
        with patch("notifier.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            with patch.dict("os.environ", {
                "TELEGRAM_BOT_TOKEN": "fake",
                "TELEGRAM_CHAT_ID": "123",
            }):
                from notifier import send_lead
                send_lead(
                    "קבוצה-ראשית",
                    "תוכן כלשהו",
                    "https://m.facebook.com/groups/test/posts/1",
                    "רלוונטי",
                    also_in=["קבוצה-2", "קבוצה-3"],
                )
                sent_text = mock_post.call_args[1]["json"]["text"]
                assert "קבוצה-ראשית" in sent_text
                assert "+ 2 קבוצות נוספות" in sent_text

    def test_send_lead_no_also_in(self):
        """send_lead ללא also_in — לא מוסיף טקסט קבוצות נוספות."""
        with patch("notifier.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            with patch.dict("os.environ", {
                "TELEGRAM_BOT_TOKEN": "fake",
                "TELEGRAM_CHAT_ID": "123",
            }):
                from notifier import send_lead
                send_lead(
                    "קבוצה-יחידה",
                    "תוכן כלשהו",
                    "https://m.facebook.com/groups/test/posts/1",
                    "רלוונטי",
                )
                sent_text = mock_post.call_args[1]["json"]["text"]
                assert "קבוצה-יחידה" in sent_text
                assert "קבוצות נוספות" not in sent_text

    def test_send_lead_empty_also_in(self):
        """send_lead עם also_in ריק — לא מוסיף טקסט קבוצות נוספות."""
        with patch("notifier.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            with patch.dict("os.environ", {
                "TELEGRAM_BOT_TOKEN": "fake",
                "TELEGRAM_CHAT_ID": "123",
            }):
                from notifier import send_lead
                send_lead(
                    "קבוצה-יחידה",
                    "תוכן",
                    "https://m.facebook.com/groups/test/posts/1",
                    "רלוונטי",
                    also_in=[],
                )
                sent_text = mock_post.call_args[1]["json"]["text"]
                assert "קבוצות נוספות" not in sent_text

    @staticmethod
    def _run_xgroup_dedup(posts):
        """סימולציה של לוגיקת cross-group dedup מ-main.py שלב 2.1."""
        _seen_hashes: dict = {}
        unique_posts = []
        for p in posts:
            ch = p.get("_content_hash", "")
            if not ch:
                p["also_in"] = []
                p["_also_in_urls"] = []
                unique_posts.append(p)
                continue
            if ch in _seen_hashes:
                primary = _seen_hashes[ch]
                dup_group = p["group"]
                if dup_group != primary["group"] and dup_group not in primary["also_in"]:
                    primary["also_in"].append(dup_group)
                    dup_url = p.get("group_url", "")
                    if dup_url:
                        primary["_also_in_urls"].append(dup_url)
            else:
                p["also_in"] = []
                p["_also_in_urls"] = []
                _seen_hashes[ch] = p
                unique_posts.append(p)
        return unique_posts

    def test_content_dedup_hash_merges_groups(self):
        """פוסטים עם אותו content_hash מאוחדים — also_in מתעדכן."""
        from main import _content_dedup_hash

        text = "שלום אני מחפש מתכנת לפרויקט גדול ומורכב במיוחד"
        c_hash = _content_dedup_hash(text)

        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": text, "_content_hash": c_hash},
            {"id": "p2", "group": "קבוצה-ב", "content": text, "_content_hash": c_hash},
            {"id": "p3", "group": "קבוצה-ג", "content": text, "_content_hash": c_hash},
        ]
        unique = self._run_xgroup_dedup(posts)

        assert len(unique) == 1
        assert unique[0]["group"] == "קבוצה-א"
        assert unique[0]["also_in"] == ["קבוצה-ב", "קבוצה-ג"]

    def test_different_content_not_merged(self):
        """פוסטים עם תוכן שונה לא מאוחדים."""
        from main import _content_dedup_hash

        text1 = "מחפש מפתח פייתון לפרויקט חדש ומעניין שלי"
        text2 = "דרושה מעצבת גרפית לעבודה קבועה במשרדים שלנו"

        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": text1, "_content_hash": _content_dedup_hash(text1)},
            {"id": "p2", "group": "קבוצה-ב", "content": text2, "_content_hash": _content_dedup_hash(text2)},
        ]
        unique = self._run_xgroup_dedup(posts)

        assert len(unique) == 2
        assert unique[0]["also_in"] == []
        assert unique[1]["also_in"] == []

    def test_empty_hash_not_merged(self):
        """פוסטים ללא hash (טקסט ריק) לא מאוחדים ביניהם."""
        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": "", "_content_hash": ""},
            {"id": "p2", "group": "קבוצה-ב", "content": "", "_content_hash": ""},
        ]
        unique = self._run_xgroup_dedup(posts)
        assert len(unique) == 2

    def test_same_group_not_added_to_also_in(self):
        """שני פוסטים מאותה קבוצה עם hash זהה — לא מוסיפים לalso_in."""
        from main import _content_dedup_hash

        text = "פוסט כפול מאותה קבוצה עם תוכן זהה לגמרי"
        c_hash = _content_dedup_hash(text)

        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": text, "_content_hash": c_hash},
            {"id": "p2", "group": "קבוצה-א", "content": text, "_content_hash": c_hash},
        ]
        unique = self._run_xgroup_dedup(posts)

        assert len(unique) == 1
        # אותה קבוצה — also_in ריק, לא מציג "קבוצות נוספות" מטעה
        assert unique[0]["also_in"] == []

    def test_also_in_urls_collected(self):
        """group_url של קבוצות שאוחדו נשמר ב-_also_in_urls — לבדיקת force_send."""
        from main import _content_dedup_hash

        text = "פוסט שפורסם בכמה קבוצות עם מילות מפתח שונות לכל קבוצה"
        c_hash = _content_dedup_hash(text)

        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": text, "_content_hash": c_hash,
             "group_url": "https://m.facebook.com/groups/a"},
            {"id": "p2", "group": "קבוצה-ב", "content": text, "_content_hash": c_hash,
             "group_url": "https://m.facebook.com/groups/b"},
            {"id": "p3", "group": "קבוצה-ג", "content": text, "_content_hash": c_hash,
             "group_url": "https://m.facebook.com/groups/c"},
        ]
        unique = self._run_xgroup_dedup(posts)

        assert len(unique) == 1
        assert unique[0]["_also_in_urls"] == [
            "https://m.facebook.com/groups/b",
            "https://m.facebook.com/groups/c",
        ]

    def test_duplicate_group_name_not_added_twice(self):
        """אותו שם קבוצה לא מתווסף ל-also_in יותר מפעם אחת."""
        from main import _content_dedup_hash

        text = "פוסט כפול שמופיע שלוש פעמים באותן שתי קבוצות"
        c_hash = _content_dedup_hash(text)

        posts = [
            {"id": "p1", "group": "קבוצה-א", "content": text, "_content_hash": c_hash},
            {"id": "p2", "group": "קבוצה-ב", "content": text, "_content_hash": c_hash},
            {"id": "p3", "group": "קבוצה-ב", "content": text, "_content_hash": c_hash},
        ]
        unique = self._run_xgroup_dedup(posts)

        assert len(unique) == 1
        # קבוצה-ב מופיעה רק פעם אחת
        assert unique[0]["also_in"] == ["קבוצה-ב"]


# ──────────────────────────────────────────────────────────────
# מילים חמות (hot words)
# ──────────────────────────────────────────────────────────────

from main import matches_hot_word, add_hot_word, remove_hot_word


class TestMatchesHotWord:
    """בדיקות לפונקציית matches_hot_word — זיהוי מילים חמות בפוסטים."""

    def test_match_found(self):
        with patch.dict("main._keywords_state", {"hot_words": ["דחוף", "מיידי"]}):
            assert matches_hot_word("צריך עזרה דחוף!") == "דחוף"

    def test_no_match(self):
        with patch.dict("main._keywords_state", {"hot_words": ["דחוף", "מיידי"]}):
            assert matches_hot_word("פוסט רגיל") is None

    def test_case_insensitive(self):
        with patch.dict("main._keywords_state", {"hot_words": ["urgent"]}):
            assert matches_hot_word("This is URGENT!") == "urgent"

    def test_empty_hot_words(self):
        """רשימת מילים חמות ריקה — לא מחזירה כלום."""
        with patch.dict("main._keywords_state", {"hot_words": []}):
            assert matches_hot_word("דחוף מאוד") is None

    def test_no_hot_words_key(self):
        """אין מפתח hot_words ב-state — לא קורס."""
        with patch.dict("main._keywords_state", {}, clear=True):
            assert matches_hot_word("דחוף") is None

    def test_first_match_returned(self):
        """מחזיר את המילה הראשונה שנמצאה."""
        with patch.dict("main._keywords_state", {"hot_words": ["דחוף", "מיידי"]}):
            assert matches_hot_word("צריך מיידי ודחוף") == "דחוף"

    def test_substring_match(self):
        """מילה חמה כ-substring — נתפסת (כמו בשאר מנגנוני הסינון)."""
        with patch.dict("main._keywords_state", {"hot_words": ["דחוף"]}):
            assert matches_hot_word("בדחיפות") is None  # "דחוף" לא ב-"בדחיפות"
            assert matches_hot_word("דחוף!!") == "דחוף"


class TestAddRemoveHotWord:
    """בדיקות ל-CRUD של מילים חמות."""

    @patch("main._save_hot_words")
    @patch("main._load_hot_words", return_value=[])
    def test_add_new_word(self, mock_load, mock_save):
        ok, msg = add_hot_word("דחוף")
        assert ok is True
        assert "נוספה" in msg
        mock_save.assert_called_once_with(["דחוף"])

    @patch("main._save_hot_words")
    @patch("main._load_hot_words", return_value=["דחוף"])
    def test_add_duplicate(self, mock_load, mock_save):
        ok, msg = add_hot_word("דחוף")
        assert ok is False
        assert "כבר קיימת" in msg
        mock_save.assert_not_called()

    @patch("main._save_hot_words")
    @patch("main._load_hot_words", return_value=[])
    def test_add_empty(self, mock_load, mock_save):
        ok, msg = add_hot_word("  ")
        assert ok is False
        mock_save.assert_not_called()

    @patch("main._save_hot_words")
    @patch("main._load_hot_words", return_value=["דחוף", "מיידי"])
    def test_remove_word(self, mock_load, mock_save):
        ok, msg = remove_hot_word("דחוף")
        assert ok is True
        assert "הוסרה" in msg
        mock_save.assert_called_once_with(["מיידי"])

    @patch("main._save_hot_words")
    @patch("main._load_hot_words", return_value=["דחוף"])
    def test_remove_nonexistent(self, mock_load, mock_save):
        ok, msg = remove_hot_word("לא קיים")
        assert ok is False
        assert "לא נמצאה" in msg
        mock_save.assert_not_called()


class TestSendLeadHotWord:
    """בדיקות ל-send_lead עם פרמטר is_hot."""

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "123"})
    def test_hot_lead_has_prefix(self, mock_post):
        """ליד חם — ההודעה מתחילה בשורת ליד חם."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        from notifier import send_lead
        send_lead("קבוצה", "תוכן", "https://m.facebook.com/groups/test/posts/123", "סיבה",
                  is_hot=True)
        call_args = mock_post.call_args
        text = call_args.kwargs.get("json", {}).get("text", "")
        assert "ליד חם" in text

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "123"})
    def test_normal_lead_no_prefix(self, mock_post):
        """ליד רגיל — ללא שורת ליד חם."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        from notifier import send_lead
        send_lead("קבוצה", "תוכן", "https://m.facebook.com/groups/test/posts/123", "סיבה",
                  is_hot=False)
        call_args = mock_post.call_args
        text = call_args.kwargs.get("json", {}).get("text", "")
        assert "ליד חם" not in text

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "123"})
    def test_hot_lead_uses_markdown(self, mock_post):
        """ליד חם — נשלח עם parse_mode Markdown."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        from notifier import send_lead
        send_lead("קבוצה", "תוכן", "https://m.facebook.com/groups/test/posts/123", "סיבה",
                  is_hot=True)
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload.get("parse_mode") == "Markdown"
        assert "ליד חם" in payload["text"]


class TestSendLeadAuthorUrl:
    """בדיקות ל-send_lead עם פרמטר author_url."""

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_author_url_shown(self, mock_post):
        """כשיש author_url — מציג קישור לפרופיל המפרסם."""
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_lead
        send_lead("קבוצה", "תוכן הפוסט כאן", "https://m.facebook.com/groups/g/posts/1",
                  "סיבה", author_url="https://m.facebook.com/profile.php?id=999")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert "פרופיל המפרסם" in payload["text"]
        # URL ממומר לדסקטופ
        assert "www.facebook.com/profile.php?id=999" in payload["text"]

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_no_author_url(self, mock_post):
        """כשאין author_url — לא מציג שורת פרופיל."""
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_lead
        send_lead("קבוצה", "תוכן הפוסט כאן", "https://m.facebook.com/groups/g/posts/1",
                  "סיבה")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert "פרופיל המפרסם" not in payload["text"]

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_empty_author_url(self, mock_post):
        """author_url ריק — לא מציג שורת פרופיל."""
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_lead
        send_lead("קבוצה", "תוכן הפוסט כאן", "https://m.facebook.com/groups/g/posts/1",
                  "סיבה", author_url="")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert "פרופיל המפרסם" not in payload["text"]


# ──────────────────────────────────────────────────────────────
# מכרזים — scraper_tenders
# ──────────────────────────────────────────────────────────────

class TestTenderIdExtraction:
    """בדיקות ל-_tender_id — חילוץ מזהה ייחודי למכרז."""

    def test_id_from_ckan_id(self):
        from scraper_tenders import _tender_id
        assert _tender_id({"_id": 12345}) == "tender_12345"

    def test_id_from_publication_number(self):
        from scraper_tenders import _tender_id
        record = {"_id": "", "מספר פרסום": "4000600977"}
        assert _tender_id(record) == "tender_4000600977"

    def test_id_fallback_hash(self):
        from scraper_tenders import _tender_id
        record = {"שם המכרז": "מכרז ניקיון", "גורם מפרסם": "משרד הבריאות"}
        tid = _tender_id(record)
        assert tid.startswith("tender_hash_")
        assert len(tid) > len("tender_hash_")

    def test_same_content_same_hash(self):
        from scraper_tenders import _tender_id
        r1 = {"שם המכרז": "ניקיון", "גורם מפרסם": "בריאות"}
        r2 = {"שם המכרז": "ניקיון", "גורם מפרסם": "בריאות"}
        assert _tender_id(r1) == _tender_id(r2)


class TestTenderIsActive:
    """בדיקות ל-_is_active — סינון מכרזים שפג תוקפם."""

    def test_future_deadline_is_active(self):
        from scraper_tenders import _is_active
        assert _is_active({"תאריך אחרון להגשה": "2099-12-31"}) is True

    def test_past_deadline_is_inactive(self):
        from scraper_tenders import _is_active
        assert _is_active({"תאריך אחרון להגשה": "2020-01-01"}) is False

    def test_no_deadline_is_active(self):
        from scraper_tenders import _is_active
        assert _is_active({}) is True

    def test_various_date_formats(self):
        from scraper_tenders import _is_active
        assert _is_active({"תאריך אחרון להגשה": "31/12/2099"}) is True
        assert _is_active({"תאריך אחרון להגשה": "01/01/2020"}) is False
        assert _is_active({"מועד אחרון להגשה": "2099-12-31 12:00:00"}) is True


class TestTenderFormatting:
    """בדיקות ל-_format_tender — המרת רשומת CKAN למבנה ליד."""

    def test_basic_formatting(self):
        from scraper_tenders import _format_tender
        record = {
            "_id": 99,
            "שם המכרז": "מכרז ניקיון",
            "גורם מפרסם": "משרד הבריאות",
            "תאריך אחרון להגשה": "2099-06-15",
            "תחום": "שירותי ניקיון",
            "מספר פרסום": "4000600977",
        }
        result = _format_tender(record)
        assert result["name"] == "מכרז ניקיון"
        assert result["publisher"] == "משרד הבריאות"
        assert "15/06/2099" in result["deadline"]
        assert result["category"] == "שירותי ניקיון"
        assert "4000600977" in result["url"]
        assert result["source"] == "mr.gov.il"

    def test_missing_fields(self):
        from scraper_tenders import _format_tender
        result = _format_tender({"_id": 1})
        assert result["name"] == "מכרז ללא שם"
        assert result["publisher"] == ""

    def test_mr_gov_url(self):
        from scraper_tenders import _build_mr_gov_url
        assert "4000600977" in _build_mr_gov_url({"מספר פרסום": "4000600977"})
        assert "TENDER" in _build_mr_gov_url({})


class TestFetchTenders:
    """בדיקות ל-fetch_tenders — חיפוש ו-dedup."""

    @patch("scraper_tenders._search_tenders")
    @patch("scraper_tenders._get_resource_id", return_value="test-resource")
    def test_dedup_across_keywords(self, mock_rid, mock_search):
        """מכרז שמופיע בשתי מילות חיפוש לא נשלח פעמיים."""
        from scraper_tenders import fetch_tenders
        record = {"_id": 1, "שם המכרז": "ניקיון", "גורם מפרסם": "בריאות"}
        mock_search.return_value = [record]
        result = fetch_tenders(["ניקיון", "תחזוקה"])
        assert len(result) == 1

    @patch("scraper_tenders._search_tenders")
    @patch("scraper_tenders._get_resource_id", return_value="test-resource")
    def test_empty_keywords_returns_empty(self, mock_rid, mock_search):
        from scraper_tenders import fetch_tenders
        result = fetch_tenders([])
        assert result == []
        mock_search.assert_not_called()

    @patch("scraper_tenders._search_tenders")
    @patch("scraper_tenders._get_resource_id", return_value="test-resource")
    def test_filters_inactive(self, mock_rid, mock_search):
        from scraper_tenders import fetch_tenders
        active = {"_id": 1, "שם המכרז": "פעיל", "תאריך אחרון להגשה": "2099-12-31"}
        expired = {"_id": 2, "שם המכרז": "פג", "תאריך אחרון להגשה": "2020-01-01"}
        mock_search.return_value = [active, expired]
        result = fetch_tenders(["ניקיון"])
        assert len(result) == 1
        assert result[0]["name"] == "פעיל"


class TestClassifyTender:
    """בדיקות לסיווג מכרזים עם AI."""

    @patch("classifier._chat_completion")
    def test_classify_tender_relevant(self, mock_chat):
        mock_chat.return_value = '{"relevant": true, "reason": "מכרז ניקיון למוסד ציבורי"}'
        from classifier import classify_tender
        result = classify_tender("שירותי ניקיון למשרד הבריאות", "משרד הבריאות", "שירותי ניקיון")
        assert result["relevant"] is True

    @patch("classifier._chat_completion")
    def test_classify_tender_not_relevant(self, mock_chat):
        mock_chat.return_value = '{"relevant": false, "reason": "מכרז לניקיון נתונים, לא ניקיון פיזי"}'
        from classifier import classify_tender
        result = classify_tender("ניקיון נתונים ממערכת IT", "משרד החינוך", "IT")
        assert result["relevant"] is False

    @patch("classifier._chat_completion")
    def test_classify_tender_batch(self, mock_chat):
        mock_chat.return_value = '[{"relevant": true, "reason": "ניקיון"}, {"relevant": false, "reason": "לא רלוונטי"}]'
        from classifier import classify_tender_batch
        tenders = [
            {"name": "ניקיון משרדים", "publisher": "עירייה", "category": "ניקיון"},
            {"name": "מכרז IT", "publisher": "משרד", "category": "טכנולוגיה"},
        ]
        results = classify_tender_batch(tenders)
        assert len(results) == 2
        assert results[0]["relevant"] is True
        assert results[1]["relevant"] is False

    @patch("classifier._chat_completion")
    def test_batch_fallback_on_bad_json(self, mock_chat):
        """JSON לא תקין בבאצ' — נופל לסיווג בודד."""
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not valid json"
            return '{"relevant": true, "reason": "ok"}'
        mock_chat.side_effect = side_effect
        from classifier import classify_tender_batch
        tenders = [{"name": "ניקיון", "publisher": "א", "category": ""}]
        results = classify_tender_batch(tenders)
        assert len(results) == 1

    def test_tender_criteria_has_default(self):
        """קריטריוני סיווג מכרזים כוללים ברירת מחדל מובנית."""
        from classifier import _get_tender_criteria
        criteria = _get_tender_criteria()
        assert "ניקיון" in criteria
        assert "מוסדות" in criteria or "ציבוריים" in criteria


class TestTenderKeywords:
    """בדיקות לטעינת מילות חיפוש מכרזים."""

    @patch.dict("os.environ", {"TENDER_KEYWORDS": "ניקיון,אבטחה,גינון"})
    def test_env_var_parsing(self):
        # צריך לטעון מחדש כי הקבוע נקבע בזמן import
        import importlib
        import scraper_tenders
        importlib.reload(scraper_tenders)
        assert "ניקיון" in scraper_tenders.TENDER_KEYWORDS_DEFAULT
        assert "אבטחה" in scraper_tenders.TENDER_KEYWORDS_DEFAULT
        assert "גינון" in scraper_tenders.TENDER_KEYWORDS_DEFAULT

    def test_load_from_db_fallback(self):
        from scraper_tenders import load_tender_keywords
        # ללא DB — חוזר ל-default
        keywords = load_tender_keywords()
        assert isinstance(keywords, list)
        assert len(keywords) > 0


class TestSendTender:
    """בדיקות ל-send_tender — פורמט הודעת מכרז בטלגרם."""

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_send_tender_format(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_tender
        result = send_tender(
            name="מכרז ניקיון 3/2025",
            publisher="רשות המסים",
            deadline="22/05/2025 12:00",
            url="https://mr.gov.il/ilgstorefront/he/p/4000600977",
            category="שירותי ניקיון",
        )
        assert result is True
        payload = mock_post.call_args.kwargs.get("json", {})
        text = payload["text"]
        assert "מכרז חדש" in text
        assert "מכרז ניקיון 3/2025" in text
        assert "רשות המסים" in text
        assert "22/05/2025" in text
        assert "שירותי ניקיון" in text
        assert "mr.gov.il" in text

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_send_tender_no_deadline(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_tender
        send_tender(name="מכרז", publisher="", deadline="",
                    url="https://example.com")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert "מועד אחרון" not in payload["text"]

    @patch("notifier.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"})
    def test_send_tender_no_publisher(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        from notifier import send_tender
        send_tender(name="מכרז", publisher="", deadline="",
                    url="https://example.com")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert "גוף מפרסם" not in payload["text"]


class TestTenderDB:
    """בדיקות ל-DB של מכרזים — seen_tenders."""

    def test_mark_and_check(self, tmp_path, monkeypatch):
        """מכרז שנשמר מזוהה כנראה."""
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        # אתחול מחדש של thread-local connection
        if hasattr(database._local, "conn"):
            del database._local.conn
        database.init_db()
        assert database.is_tender_seen("tender_123") is False
        database.mark_tender_sent("tender_123", "מכרז ניקיון", "משרד הבריאות")
        assert database.is_tender_seen("tender_123") is True

    def test_duplicate_insert(self, tmp_path, monkeypatch):
        """INSERT כפול לא זורק שגיאה (INSERT OR IGNORE)."""
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        if hasattr(database._local, "conn"):
            del database._local.conn
        database.init_db()
        database.mark_tender_sent("tender_456", "מכרז א", "גוף א")
        database.mark_tender_sent("tender_456", "מכרז א", "גוף א")
        assert database.is_tender_seen("tender_456") is True

    def test_stats(self, tmp_path, monkeypatch):
        import database
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
        if hasattr(database._local, "conn"):
            del database._local.conn
        database.init_db()
        database.mark_tender_sent("t1", "א", "x")
        database.mark_tender_sent("t2", "ב", "y")
        stats = database.get_tender_stats()
        assert stats["total_tenders_sent"] == 2
