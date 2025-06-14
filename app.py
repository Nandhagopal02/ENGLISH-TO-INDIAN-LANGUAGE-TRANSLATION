import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from gtts import gTTS
import os

# Load model
@st.cache_resource
def load_model():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return tokenizer, model

tokenizer, model = load_model()

# Language map
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
st.title("🌐 English to Indian Language Translator with Voice")
text = st.text_area("Enter English text here:")

lang = st.selectbox("Choose a target language", list(indian_languages.keys()))

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        try:
            tokenizer.src_lang = "en_XX"
            inputs = tokenizer(text, return_tensors="pt")
            tgt_lang_code = indian_languages[lang]
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
            )
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            st.success(f"✅ Translation in {lang}:")
            st.write(translation)

            # Voice Output
            tts = gTTS(translation, lang=tgt_lang_code[:2])
            tts.save("output.mp3")
            st.audio("output.mp3")
            os.remove("output.mp3")

        except Exception as e:
            st.error(f"Translation failed: {e}")
