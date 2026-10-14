import streamlit as st
import re, soundfile as sf, numpy as np
from transformers import pipeline
import torch

st.set_page_config(page_title="Bambara TTS Pro", page_icon="🎤")
st.title("🎤 Bambara TTS Pro")
st.markdown("**100=kɛmɛ | 7=wolonwula | G7=Gwolonwula | 4nan=naaninan | 4,8=naani n'a kunkanfɛn seegin | 2027=bafila ani mugan ni wolonwula**")

@st.cache_resource
def load_tts():
    return pipeline("text-to-speech", model="facebook/mms-tts-bam")

tts = load_tts()

def num_to_bam(n):
    n=int(n)
    d={1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seegin",9:"kɔnɔntɔn",10:"tan",20:"mugan",30:"bisaba",100:"kɛmɛ"}
    if n in d: return d[n]
    if n==2027: return "bafila ani mugan ni wolonwula"
    if n==2026: return "bafila ani mugan ni wɔɔrɔ"
    if n==2029: return "bafila ani mugan ni kɔnɔntɔn"
    if n==2030: return "bafila ani bisaba"
    if n<100:
        t=(n//10)*10; r=n%10
        return d[t] if r==0 else f"{d[t]} ni {d[r]}"
    if n<1000:
        return f"{d[n//100]} kɛmɛ" if n%100==0 else f"{d[n//100]} kɛmɛ ni {num_to_bam(n%100)}"
    if n<10000:
        ba=n//1000; rest=n%1000
        base="bafila" if ba==2 else f"ba {d[ba]}"
        return base if rest==0 else f"{base} ani {num_to_bam(rest)}"
    return str(n)

def normalize(text):
    text=re.sub(r'([A-Za-z])(\d+)', lambda m: f"{m.group(1)} {num_to_bam(m.group(2))}", text)
    text=re.sub(r'(\d+)nan', lambda m: f"{num_to_bam(m.group(1))}nan", text)
    text=re.sub(r'(\d+)[,\.](\d+)', lambda m: f"{num_to_bam(m.group(1))} n'a kunkanfɛn {num_to_bam(m.group(2))}", text)
    text=re.sub(r'\d+', lambda m: num_to_bam(m.group(0)), text)
    return text

text = st.text_area("Bamanankan sɛbɛn yan - مهما طال النص", "Aw ni ce, 2027 ye bafila ani mugan ni wolonwula ye, 100 ye kɛmɛ ye, G7 ye Gwolonwula ye, 4nan, 4,8", height=150)

if st.button("Fɔ! - توليد 🎧", type="primary"):
    clean = normalize(text)
    st.write(f"**Nɔrɔli kɔfɛ:** {clean}")
    sents = re.split(r'[.!?\n]+', clean)
    audios=[]; sr=16000
    with st.spinner("A bɛ fɔ..."):
        for s in sents:
            if not s.strip(): continue
            for i in range(0, len(s), 150):
                c=s[i:i+150].strip()
                if not c: continue
                r=tts(c)
                audios.append(r["audio"]); sr=r["sampling_rate"]
        final = np.concatenate(audios) if audios else np.array([0])
        sf.write("/tmp/b.wav", final, sr)
        st.audio("/tmp/b.wav")
        st.success("✅ A banna! Da saniyan, garagara tɛ - harf bɛɛ bɛ bɔ a yɔrɔ la!")
