
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from gtts import gTTS
import torch
import os

# Load model & tokenizer
@st.cache_resource
def load_model():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    return tokenizer, model

tokenizer, model = load_model()

# Indian language code map
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

# Streamlit UI
st.title("🇮🇳 English to Indian Language Translator with Voice")
st.write("Translate English → Indian Languages and hear the result spoken aloud.")

text_input = st.text_area("✍️ Enter English text to translate", height=150)
target_language = st.selectbox("🌐 Choose your target Indian language", list(indian_languages.keys()))

if st.button("Translate"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Translation
        target_lang_code = indian_languages[target_language]
        inputs = tokenizer(text_input, return_tensors="pt")
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # Show text output
        st.success(f"🈯 Translation in {target_language}:")
        st.write(translated_text)

        # Generate speech with gTTS
        try:
            tts = gTTS(text=translated_text, lang=target_lang_code[:2])  # e.g., 'ta' from 'ta_IN'
            audio_file = f"tts_{target_lang_code}.mp3"
            tts.save(audio_file)

            # Play audio
            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")

            # Clean up
            os.remove(audio_file)
        except Exception as e:
            st.error(f"❌ TTS failed: {e}")
            st.info("Not all languages are supported by gTTS. Try Hindi, Tamil, etc.")
