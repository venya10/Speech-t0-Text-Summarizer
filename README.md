# Speech-to-Text Summarizer

A Streamlit web app that transcribes audio using OpenAI Whisper, summarizes the transcription using BART, and lets you ask questions about the content via a Llama-3.3-70B chatbot powered by Groq.

## Features

- **Transcription** — Upload `.wav`, `.mp3`, or `.mp4` audio and get a full text transcription
- **Summarization** — Automatically summarize the transcription using `facebook/bart-large-cnn`
- **Q&A Chatbot** — Ask questions about the transcription and summary using Llama-3.3-70B via Groq

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/venya10/Speech-to-Text-Summarizer.git
cd Speech-to-Text-Summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

## Requirements

- Python 3.9+
- GPU recommended for larger Whisper models; CPU works fine for `tiny` and `base`

## Models

| Task | Model |
|------|-------|
| Transcription | [OpenAI Whisper](https://github.com/openai/whisper) (selectable size: tiny → large) |
| Summarization | [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) |
| Q&A Chatbot | Llama-3.3-70B-Versatile via [Groq](https://console.groq.com) |
