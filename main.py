import os
import streamlit as st
from groq import Groq
from moviepy import *
from pydub import AudioSegment

# Função para extrair o áudio do vídeo
def extract_audio(video_file):
    st.info("Extraindo áudio do vídeo...")
    video = moviepy.VideoFileClip(video_file)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    st.success("Áudio extraído com sucesso.")
    return audio_path

# Função para dividir o áudio em chunks
def split_audio(audio_path, chunk_length_ms=30000):
    st.info("Dividindo o áudio em chunks...")
    audio = AudioSegment.from_file(audio_path, format="wav")
    chunks = [audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    st.success(f"Áudio dividido em {len(chunks)} chunks.")
    return chunks

# Função para salvar cada chunk como arquivo separado
def save_chunks(chunks):
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    return chunk_files

# Função para transcrição usando Groq
def transcribe_chunk(audio_path, client):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="pt",  # Ajuste para o idioma desejado
            temperature=0.0
        )
    return transcription.text

# Pipeline principal
def main():
    st.title("Transcrição de Vídeo com Groq")
    st.info("Faça upload do seu arquivo de vídeo para iniciar o processo.")

    # Upload do vídeo
    uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Salvar vídeo temporariamente
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extrair áudio
        audio_path = extract_audio(video_path)

        # Dividir o áudio em chunks
        chunks = split_audio(audio_path)
        chunk_files = save_chunks(chunks)

        # Configurar o cliente do Groq usando variável de ambiente
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("API Key não configurada! Configure a variável de ambiente 'GROQ_API_KEY'.")
            return
        
        client = Groq(api_key=api_key)

        # Transcrever cada chunk
        st.info("Iniciando a transcrição...")
        full_transcription = ""
        for i, chunk_file in enumerate(chunk_files):
            st.write(f"Transcrevendo chunk {i+1}/{len(chunk_files)}...")
            transcription = transcribe_chunk(chunk_file, client)
            full_transcription += transcription + "\n"
            os.remove(chunk_file)  # Remover chunk processado

        # Exibir e salvar a transcrição completa
        st.subheader("Transcrição Completa:")
        st.text_area("Resultado", full_transcription, height=300)

        # Salvar como arquivo de texto
        with open("transcription.txt", "w") as f:
            f.write(full_transcription)
        st.success("Transcrição salva como 'transcription.txt'.")
        st.download_button("Baixar Transcrição", data=full_transcription, file_name="transcription.txt")

        # Limpar arquivos temporários
        os.remove(video_path)
        os.remove(audio_path)

if __name__ == "__main__":
    main()
