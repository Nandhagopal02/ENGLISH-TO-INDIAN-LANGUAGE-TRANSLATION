import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from gtts import gTTS
import os

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    return tokenizer, model

tokenizer, model = load_model()

# Language codes
indian_languages = {
    "Hindi": "hi_IN",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Bengali": "bn_IN",
    "Gujarati": "gu_IN",
    "Kannada": "kn_IN",
    "Malayalam": "ml_IN",
    "Marathi": "mr_IN",
    "Punjabi": "pa_IN",
    "Urdu": "ur_IN"
}

# UI
st.title("🇮🇳 English to Indian Language Translator with Voice")
text = st.text_area("Enter English text:")

lang = st.selectbox("Choose a target language:", list(indian_languages.keys()))

if st.button("Translate"):
    if not text.strip():
        st.warning("Please type something to translate.")
    else:
        tgt_lang_code = indian_languages[lang]
        encoded = tokenizer(text, return_tensors="pt")
        translated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
        )
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success(f"Translation in {lang}:")
        st.write(result)

        # Voice output
        try:
            tts = gTTS(result, lang=tgt_lang_code[:2])
            tts.save("speech.mp3")
            st.audio("speech.mp3", format="audio/mp3")
            os.remove("speech.mp3")
        except Exception as e:
            st.error(f"Speech error: {e}")
