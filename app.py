import streamlit as st
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile

# page setup
st.set_page_config(page_title="Speech-to-Text Summarization", layout="wide")
st.title("Speech-to-Text Summarization")
st.write("Upload an audio file, transcribe it using Whisper, and summarize it using BART.")

#load Whisper model

@st.cache_resource
def load_whisper_model(size="base"):
    return whisper.load_model(size)

whisper_model_size = st.sidebar.selectbox(
    "Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=1
)
st.sidebar.text("Larger models are slower but more accurate.")
whisper_model = load_whisper_model(whisper_model_size)


#load BART model
@st.cache_resource
def load_bart_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, bart_model, device = load_bart_model()

#upload audio

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "mp4"])

if audio_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(audio_file.read())
    tfile_path = tfile.name

    st.info("Transcribing audio with Whisper...")
    result = whisper_model.transcribe(tfile_path)
    transcription = result["text"]
    st.subheader("Transcription")
    st.write(transcription)


    #summarize 
   
    st.info("Generating summary with BART...")
    inputs = tokenizer(transcription, return_tensors="pt", truncation=True, max_length=1024).to(device)

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        min_length=30,
        num_beams=6,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        forced_bos_token_id=bart_model.config.bos_token_id
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.subheader("Summary")
    st.write(summary)

    # copy/download buttons
    st.download_button("Download Transcription", transcription, file_name="transcription.txt")
    st.download_button("Download Summary", summary, file_name="summary.txt")



# CHATBOT (LLAMA-3.3-70B-VERSATILE)
import os

from dotenv import load_dotenv
from groq import Groq

# ==================================================
# ENV + GROQ CLIENT
# ==================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY in .env file")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

st.divider()
st.subheader("💬 Ask Questions about the Audio")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input(
    "Ask a question about the transcription or summary"
)

def ask_llama(question, transcription, summary):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n\n"
                    "Answer using the transcription and summary below.\n\n"
                    f"TRANSCRIPTION:\n{transcription}\n\n"
                    f"SUMMARY:\n{summary}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content

if user_question and transcription:
    st.session_state.chat_history.append(("user", user_question))

    with st.spinner("🤖 Thinking..."):
        answer = ask_llama(user_question, transcription, summary)

    st.session_state.chat_history.append(("assistant", answer))

# ==================================================
# DISPLAY CHAT
# ==================================================
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)