import streamlit as st
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from io import BytesIO
import uuid
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_generators():
    return {
        "GPT-2": pipeline("text-generation", model="gpt2"),
        "GPT-2 Medium": pipeline("text-generation", model="gpt2-medium"),
        "GPT-Neo 125M": pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    }

@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_boost_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def get_image_caption(image):
    processor, model = load_caption_model()
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_story(generator, prompt):
    output = generator(
        prompt,
        max_length=1024,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=50256
    )
    return output[0]['generated_text']

def score_story(story, tone_label, all_stories):
    tone_keywords = {
        "dramatic": ["suddenly", "unfortunately", "shattered", "rage"],
        "humorous": ["silly", "awkward", "lol", "giggle"],
        "poetic": ["whispers", "eternity", "moonlight", "dreams"]
    }
    tone_score = sum(word in story.lower() for word in tone_keywords.get(tone_label, [])) / 5
    readability = flesch_reading_ease(story)
    tfidf = TfidfVectorizer().fit_transform([story] + all_stories)
    sim_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    diversity = 1 - np.max(sim_scores) if len(sim_scores) > 0 else 1.0
    return (0.35 * tone_score) + (0.25 * readability) + (0.4 * diversity)

def boost_story(story, tone):
    booster = load_boost_model()
    prompt = f"Refine this {tone} story for vivid imagery and strong structure:\n\n{story}"
    return booster(prompt, max_length=1024, do_sample=True)[0]['generated_text']

def narrate_text_gtts_bytes(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="StoryCrafter AI", layout="wide")
st.title("📝 StoryCrafter AI")
st.caption("✨ Generate stories from multiple models, apply bagging to select the best, and refine it through boosting!")

uploaded_image = st.file_uploader("📷 Upload an image (optional)", type=["jpg", "png"])
user_input = st.text_area("💡 Describe your idea", height=100)
tone = st.selectbox("🎭 Choose a tone", ["dramatic", "humorous", "poetic"])
tone_desc = {
    "dramatic": "Write a long dramatic story about:",
    "humorous": "Write a long funny story about:",
    "poetic": "Write a long poetic narrative about:"
}[tone]

# Init session
st.session_state.setdefault("generated_stories", [])
st.session_state.setdefault("best_story", "")
st.session_state.setdefault("boosted_story", "")

if st.button("🧠 Generate from All Models"):
    if not user_input.strip() and not uploaded_image:
        st.warning("Please provide a prompt or upload an image.")
    else:
        caption = ""
        if uploaded_image:
            with st.spinner("Generating image caption..."):
                image = Image.open(uploaded_image)
                caption = get_image_caption(image)
                st.image(image, caption=f"🖼 Caption: {caption}")

        final_prompt = f"{tone_desc} {caption}. {user_input}"
        st.session_state.generated_stories.clear()

        with st.spinner("Generating stories..."):
            for model_name, gen in load_generators().items():
                story = generate_story(gen, final_prompt)
                st.session_state.generated_stories.append((model_name, story))

# Show outputs
if st.session_state.generated_stories:
    st.markdown("### 📚 Outputs from Each Model")
    for model_name, story in st.session_state.generated_stories:
        st.subheader(f"{model_name}")
        st.text_area(f"{model_name} Output", value=story, height=300, key=model_name)

# Bagging
if st.session_state.generated_stories and st.button("📊 Apply Bagging"):
    with st.spinner("Scoring stories..."):
        all = [s[1] for s in st.session_state.generated_stories]
        scored = [(m, s, score_story(s, tone, all)) for m, s in st.session_state.generated_stories]
        best = max(scored, key=lambda x: x[2])
        st.session_state.best_story = best[1]
        st.success(f"✅ Best story is from **{best[0]}**, Score: {best[2]:.2f}")
        st.text_area("🏆 Best Story", value=best[1], height=300)

# Boosting
if st.session_state.best_story and st.button("🚀 Apply Boosting"):
    with st.spinner("Refining with FLAN-T5..."):
        boosted = boost_story(st.session_state.best_story, tone)
        st.session_state.boosted_story = boosted
        st.text_area("🔮 Boosted Story", value=boosted, height=350)

# Narration
if st.session_state.boosted_story:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Boosted Story", data=st.session_state.boosted_story,
                           file_name="storycrafter.txt", mime="text/plain")
    with col2:
        if st.button("🔊 Listen to Story"):
            st.audio(narrate_text_gtts_bytes(st.session_state.boosted_story), format="audio/mp3")