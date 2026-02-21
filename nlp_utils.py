"""
nlp_utils.py – Multilingual NLP Translation
=============================================
Translates the AI assessment report into native Indian languages.

Supported languages:
    en  – English (no translation)
    hi  – Hindi
    ta  – Tamil
    te  – Telugu
    kn  – Kannada
    ml  – Malayalam
    bn  – Bengali
    mr  – Marathi
    gu  – Gujarati
    pa  – Punjabi

Integration options (choose one):
──────────────────────────────────
OPTION A – Google Cloud Translation API (recommended for production):
    from google.cloud import translate_v2 as gt
    client = gt.Client()
    result = client.translate(text, target_language=lang_code)
    return result["translatedText"]

OPTION B – Deep Translate (free tier, RapidAPI):
    import httpx
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, ...}
    resp = httpx.post("https://deep-translate1.p.rapidapi.com/language/translate/v2",
                      json={"q": text, "source": "en", "target": lang_code})
    return resp.json()["data"]["translations"]["translatedText"]

OPTION C – LibreTranslate (self-hosted, free):
    import httpx
    resp = httpx.post("http://localhost:5000/translate",
                      json={"q": text, "source": "en", "target": lang_code, "format": "text"})
    return resp.json()["translatedText"]

OPTION D – Sarvam.ai (India-specific, best for Indian languages):
    import httpx
    resp = httpx.post("https://api.sarvam.ai/translate",
                      headers={"API-Subscription-Key": SARVAM_API_KEY},
                      json={"input": text, "source_language_code": "en-IN",
                            "target_language_code": f"{lang_code}-IN"})
    return resp.json()["translated_text"]

This file currently provides a hardcoded phrase dictionary for offline
demo. Swap `_api_translate()` with any of the above options.
"""

import logging
import asyncio
import os

log = logging.getLogger(__name__)

# Language metadata
SUPPORTED_LANGUAGES = {
    "en": {"name": "English",    "script": "Latin"},
    "hi": {"name": "हिंदी",       "script": "Devanagari"},
    "ta": {"name": "தமிழ்",       "script": "Tamil"},
    "te": {"name": "తెలుగు",      "script": "Telugu"},
    "kn": {"name": "ಕನ್ನಡ",       "script": "Kannada"},
    "ml": {"name": "മലയാളം",    "script": "Malayalam"},
    "bn": {"name": "বাংলা",       "script": "Bengali"},
    "mr": {"name": "मराठी",       "script": "Devanagari"},
    "gu": {"name": "ગુજરાતી",     "script": "Gujarati"},
    "pa": {"name": "ਪੰਜਾਬੀ",      "script": "Gurmukhi"},
}

# ── Phrase dictionary for offline demo ───────────────────────────────────────
# In production this whole block is replaced by a real API call.
_PHRASE_MAP = {
    "AI Visual Assessment Report": {
        "hi": "एआई दृश्य मूल्यांकन रिपोर्ट",
        "ta": "AI காட்சி மதிப்பீட்டு அறிக்கை",
        "te": "AI దృశ్య మూల్యాంకన నివేదిక",
        "kn": "AI ದೃಶ್ಯ ಮೌಲ್ಯಮಾಪನ ವರದಿ",
        "ml": "AI ദൃശ്യ മൂല്യനിർണ്ണയ റിപ്പോർട്ട്",
        "bn": "এআই ভিজ্যুয়াল মূল্যায়ন প্রতিবেদন",
    },
    "This is an AI Visual Assessment only and does NOT replace medical diagnosis.": {
        "hi": "यह केवल एक एआई दृश्य मूल्यांकन है और चिकित्सा निदान की जगह नहीं लेता।",
        "ta": "இது ஒரு AI காட்சி மதிப்பீடு மட்டுமே மற்றும் மருத்துவ நோயறிதலை மாற்றாது.",
        "te": "ఇది AI దృశ్య మూల్యాంకనం మాత్రమే మరియు వైద్య నిర్ధారణను భర్తీ చేయదు.",
        "kn": "ಇದು AI ದೃಶ್ಯ ಮೌಲ್ಯಮಾಪನ ಮಾತ್ರ ಮತ್ತು ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವನ್ನು ಬದಲಿಸಲಾಗುವುದಿಲ್ಲ.",
        "ml": "ഇത് ഒരു AI ദൃശ്യ മൂല്യനിർണ്ണയം മാത്രമാണ്, ഇത് വൈദ്യ നിർണ്ണയത്തെ മാറ്റില്ല.",
        "bn": "এটি শুধুমাত্র একটি এআই ভিজ্যুয়াল মূল্যায়ন এবং চিকিৎসা নির্ণয়ের বিকল্প নয়।",
    },
}

DISCLAIMER_TRANSLATIONS = {
    "hi": "⚠️ चिकित्सा अस्वीकरण: यह एआई विश्लेषण केवल सूचनात्मक उद्देश्यों के लिए है। "
          "किसी भी स्वास्थ्य समस्या के लिए कृपया एक योग्य चिकित्सक से परामर्श लें। "
          "आपातकाल में 112 पर कॉल करें।",
    "ta": "⚠️ மருத்துவ மறுப்பு: இந்த AI பகுப்பாய்வு தகவல் நோக்கங்களுக்காக மட்டுமே. "
          "எந்த உடல்நல கவலைகளுக்கும் தகுதிவாய்ந்த மருத்துவரை அணுகவும். "
          "அவசரகாலத்தில் 112 என்று அழைக்கவும்.",
    "te": "⚠️ వైద్య నిరాకరణ: ఈ AI విశ్లేషణ సమాచార ప్రయోజనాల కోసం మాత్రమే. "
          "ఏదైనా ఆరోగ్య సమస్యలకు అర్హత కలిగిన వైద్యుడిని సంప్రదించండి. "
          "అత్యవసర పరిస్థితిలో 112 కి కాల్ చేయండి.",
    "kn": "⚠️ ವೈದ್ಯಕೀಯ ನಿರಾಕರಣೆ: ಈ AI ವಿಶ್ಲೇಷಣೆ ಮಾಹಿತಿ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ. "
          "ಯಾವುದೇ ಆರೋಗ್ಯ ಸಮಸ್ಯೆಗಳಿಗೆ ಅರ್ಹ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ. "
          "ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ 112 ಗೆ ಕರೆ ಮಾಡಿ.",
    "ml": "⚠️ വൈദ്യ നിരാകരണം: ഈ AI വിശകലനം വിവര ആവശ്യങ്ങൾക്ക് മാത്രമാണ്. "
          "ഏതെങ്കിലും ആരോഗ്യ ആശങ്കകൾക്ക് യോഗ്യനായ ഒരു ഡോക്ടറെ സമീപിക്കുക. "
          "അടിയന്തര ഘട്ടത്തിൽ 112 ൽ വിളിക്കുക.",
    "bn": "⚠️ চিকিৎসা দাবিত্যাগ: এই AI বিশ্লেষণ শুধুমাত্র তথ্যমূলক উদ্দেশ্যে। "
          "যেকোনো স্বাস্থ্য উদ্বেগের জন্য একজন যোগ্য চিকিৎসকের সাথে পরামর্শ করুন। "
          "জরুরি অবস্থায় 112 কল করুন।",
}


async def _api_translate(text: str, target_lang: str) -> str:
    """
    ── SWAP THIS FUNCTION WITH A REAL TRANSLATION API ──
    Currently returns a prefixed string to indicate translation would occur here.
    Replace with Google Translate, LibreTranslate, Sarvam.ai, etc.
    """
    # Simulate async network call latency
    await asyncio.sleep(0.05)

    # Check our phrase map first for key phrases
    for phrase_en, translations in _PHRASE_MAP.items():
        if phrase_en in text and target_lang in translations:
            text = text.replace(phrase_en, translations[target_lang])

    lang_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang)
    return (
        f"[{lang_name} Translation – Connect API]\n\n"
        f"{text}\n\n"
        f"── To enable real translation, set TRANSLATION_API_KEY in .env ──"
    )


async def translate_report(report_text: str, target_lang: str) -> dict:
    """
    Translate the full report and disclaimer.

    Args:
        report_text : English report string from image_pipeline.
        target_lang : ISO 639-1 language code.

    Returns:
        dict with translated report, disclaimer, and language metadata.
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        log.warning(f"Unsupported language '{target_lang}', defaulting to English.")
        target_lang = "en"

    if target_lang == "en":
        return {
            "language_code": "en",
            "language_name": "English",
            "report":        report_text,
            "disclaimer":    "⚠️ MEDICAL DISCLAIMER: This is an AI Visual Assessment only. "
                             "Consult a licensed medical professional. Emergency: call 112.",
            "api_status":    "passthrough",
        }

    # Translate both report and disclaimer concurrently
    translated_report, disclaimer = await asyncio.gather(
        _api_translate(report_text, target_lang),
        asyncio.sleep(0),  # placeholder for disclaimer translation
    )

    # Use hardcoded disclaimer translation if available, else English
    disc = DISCLAIMER_TRANSLATIONS.get(
        target_lang,
        "⚠️ MEDICAL DISCLAIMER: This is an AI Visual Assessment only. "
        "Consult a licensed medical professional. Emergency: call 112."
    )

    return {
        "language_code": target_lang,
        "language_name": SUPPORTED_LANGUAGES[target_lang]["name"],
        "script":        SUPPORTED_LANGUAGES[target_lang]["script"],
        "report":        translated_report,
        "disclaimer":    disc,
        "api_status":    "mock_demo",
    }


def get_supported_languages() -> list:
    """Return list of supported languages for the UI dropdown."""
    return [
        {"code": code, "name": meta["name"], "script": meta["script"]}
        for code, meta in SUPPORTED_LANGUAGES.items()
    ]
