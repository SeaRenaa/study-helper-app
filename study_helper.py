import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from transformers import pipeline
import pyttsx3
from gtts import gTTS
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import random
import json
import time
import os
from graphviz import Digraph
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Smart Study App", layout="wide")
st.title("📘 Smart Study App 📘")

@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-base")
        translator = pipeline("translation_en_to_fr", model="t5-base")
        explainer = pipeline("text2text-generation", model="google/flan-t5-base")
        quizgen = pipeline("text2text-generation", model="google/flan-t5-base")
        return summarizer, translator, explainer, quizgen
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        raise

summarizer, translator, explainer, quizgen = load_models()

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "review_stats" not in st.session_state:
    st.session_state.review_stats = {"easy": 0, "medium": 0, "hard": 0}
if "history" not in st.session_state:
    st.session_state.history = []

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join([page.extract_text() for page in reader.pages if page.extract_text()])

def summarize_text(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def translate_summary(summary):
    return translator(summary)[0]['translation_text']

def text_to_speech(summary, engine="gtts", lang="en"):
    if engine.lower() == "gtts":
        audio_path = "speech.mp3"
        try:
            if len(summary) > 4500:
                chunks = [summary[i:i+4500] for i in range(0, len(summary), 4500)]
                full_audio = BytesIO()
                for idx, chunk in enumerate(chunks):
                    tts = gTTS(text=chunk, lang=lang)
                    chunk_path = f"chunk_{idx}.mp3"
                    tts.save(chunk_path)
                    with open(chunk_path, "rb") as f:
                        full_audio.write(f.read())
                    os.remove(chunk_path)
                with open(audio_path, "wb") as final:
                    final.write(full_audio.getvalue())
            else:
                tts = gTTS(text=summary, lang=lang)
                tts.save(audio_path)
        except Exception as e:
            st.warning(f"gTTS Error: {e}")
            return None
    elif engine.lower() == "pyttsx3":
        audio_path = "speech.wav"
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.save_to_file(summary, audio_path)
            tts_engine.runAndWait()
        except Exception as e:
            st.warning(f"pyttsx3 Error: {e}")
            return None
    else:
        st.warning("Unsupported TTS engine selected.")
        return None

    return audio_path

def create_flashcards(summary):
    cards = []
    for sentence in summary.split("."):
        parts = sentence.strip().split(" is ")
        if len(parts) == 2:
            question = f"What is {parts[0]}?"
            answer = parts[1]
            cards.append({"question": question, "answer": answer})
    return cards

def feynman_explainer(summary):
    prompt = f"Explain in simple terms:\n{summary}"
    return explainer(prompt, max_length=256, do_sample=False)[0]['generated_text']

def generate_quiz(summary):
    prompt = f"Create 5 quiz questions from this summary:\n{summary}"
    return quizgen(prompt, max_length=512, do_sample=False)[0]["generated_text"]

def review_flashcards():
    if not st.session_state.flashcards:
        st.info("No flashcards yet.")
        return
    card = random.choice(st.session_state.flashcards)
    st.write(f"**Q:** {card['question']}")
    if st.button("Show Answer"):
        st.write(f"**A:** {card['answer']}")
        difficulty = st.radio("Rate difficulty:", ["easy", "medium", "hard"])
        if st.button("Next Card"):
            st.session_state.review_stats[difficulty] += 1
            st.experimental_rerun()

def display_stats():
    stats = st.session_state.review_stats
    st.write(f"🟢 Easy: {stats['easy']} | 🟡 Medium: {stats['medium']} | 🔴 Hard: {stats['hard']}")

def keyword_search(text, keyword):
    return [line for line in text.splitlines() if keyword.lower() in line.lower()]

def plot_word_frequency(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if len(word) > 3:
            freq[word] = freq.get(word, 0) + 1
    sorted_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.figure(figsize=(10, 4))
    plt.bar(sorted_freq.keys(), sorted_freq.values())
    st.pyplot(plt)

def draw_mind_map(key_ideas):
    dot = Digraph()
    dot.node("Summary", "Summary")
    for i, idea in enumerate(key_ideas):
        node = f"Idea{i}"
        dot.node(node, idea.strip()[:30] + "...")
        dot.edge("Summary", node)
    st.graphviz_chart(dot)

def save_session(summary, translated, flashcards):
    record = {
        "timestamp": time.ctime(),
        "summary": summary,
        "translated": translated,
        "flashcards": flashcards
    }
    st.session_state.history.append(record)
    try:
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")

def load_history():
    if os.path.exists("history.json"):
        try:
            with open("history.json", "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except Exception as e:
            st.warning(f"Could not load history: {e}")

load_history()

pdf = st.file_uploader("📤 Upload PDF", type="pdf")
keyword = st.text_input("🔍 Search Keyword (optional)")

if pdf:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(pdf)

        if keyword:
            matches = keyword_search(text, keyword)
            st.subheader("🔍 Keyword Matches")
            for m in matches:
                st.write("•", m)

        summary = summarize_text(text)
        st.subheader("📄 Summary")
        st.write(summary)

        if st.checkbox("🌍 Translate to Russian"):
            translated = translate_summary(summary)
            st.subheader("🇷🇺 Translated Summary")
            st.write(translated)
        else:
            translated = ""

        st.subheader("🗣 Text-to-Speech")
        engine = st.radio("Select TTS Engine", ["gTTS", "pyttsx3"])
        lang = st.radio("Select Language", ["en", "ru"])
        audio_file = text_to_speech(summary, engine=engine.lower(), lang=lang)
        if audio_file:
            st.audio(audio_file)
        else:
            st.error("Failed to generate audio.")

        st.subheader("🧠 Flashcards")
        st.session_state.flashcards = create_flashcards(summary)
        for fc in st.session_state.flashcards:
            st.write(f"• **Q:** {fc['question']}  \n  **A:** {fc['answer']}")

        st.subheader("📈 Word Frequency")
        plot_word_frequency(summary)

        st.subheader("🗺 Mind Map")
        key_ideas = summary.split(".")[:5]
        draw_mind_map(key_ideas)

        st.subheader("🧠 Explain Like I’m 5")
        if st.button("Explain Simply"):
            st.success(feynman_explainer(summary))

        st.subheader("❓ Generate Quiz")
        if st.button("Make Quiz"):
            st.code(generate_quiz(summary), language="markdown")

        st.subheader("📆 Spaced Repetition")
        review_flashcards()
        display_stats()

        if st.button("💾 Save This Session"):
            save_session(summary, translated, st.session_state.flashcards)
            st.success("✅ Saved!")

st.sidebar.header("🕓 History")
for entry in st.session_state.history[-3:]:
    st.sidebar.write(f"- {entry['timestamp']}"))
