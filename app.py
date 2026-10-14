import streamlit as st, re, soundfile as sf, numpy as np, os, tempfile
from transformers import pipeline

st.set_page_config(page_title="Bambara TTS Pro", page_icon="🎤")
st.title("🎤 Bambara TTS Pro - Sabati")

@st.cache_resource
def load_mms():
    return pipeline("text-to-speech", model="facebook/mms-tts-bam")
mms = load_mms()

def num_to_bam(n):
    n=int(n); d={1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seegin",9:"kɔnɔntɔn",10:"tan",20:"mugan",30:"bisaba",100:"kɛmɛ"}
    if n in d: return d[n]
    if n==2027: return "bafila ani mugan ni wolonwula"
    if n==2026: return "bafila ani mugan ni wɔɔrɔ"
    if n==2029: return "bafila ani mugan ni kɔnɔntɔn"
    if n==2030: return "bafila ani bisaba"
    if n<100: t=(n//10)*10; r=n%10; return d[t] if r==0 else f"{d[t]} ni {d[r]}"
    if n<1000: return f"{d[n//100]} kɛmɛ" if n%100==0 else f"{d[n//100]} kɛmɛ ni {num_to_bam(n%100)}"
    if n<10000: ba=n//1000; rest=n%1000; base="bafila" if ba==2 else f"ba {d[ba]}"; return base if rest==0 else f"{base} ani {num_to_bam(rest)}"
    return str(n)

def normalize(text):
    text=re.sub(r'([A-Za-z])(\d+)', lambda m: f"{m.group(1)} {num_to_bam(m.group(2))}", text)
    text=re.sub(r'(\d+)nan', lambda m: f"{num_to_bam(m.group(1))}nan", text)
    text=re.sub(r'(\d+)[,\.](\d+)', lambda m: f"{num_to_bam(m.group(1))} n'a kunkanfɛn {num_to_bam(m.group(2))}", text)
    text=re.sub(r'\d+', lambda m: num_to_bam(m.group(0)), text)
    return text

def tts_long(text, out="/tmp/b.wav"):
    clean=normalize(text); sents=re.split(r'[.!?\n]+', clean); audios=[]; sr=16000
    for s in sents:
        if not s.strip(): continue
        for i in range(0,len(s),150):
            c=s[i:i+150].strip()
            if not c: continue
            r=mms(c); audios.append(r["audio"]); sr=r["sampling_rate"]
    final=np.concatenate(audios) if audios else np.array([0]); sf.write(out, final, sr); return out, clean

tab1, tab2 = st.tabs(["📖 Sɛbɛn kalan", "🎭 Cloning léger"])

with tab1:
    txt = st.text_area("Bamanankan", "Aw ni ce, 2027 ye bafila ani mugan ni wolonwula ye, 100 ye kɛmɛ ye, G7 ye Gwolonwula ye, 4nan, 4,8", height=180)
    if st.button("Fɔ! 🎧", type="primary"):
        out,clean = tts_long(txt, "/tmp/mms.wav")
        st.write(clean); st.audio(out); st.success("✅ Da saniyan!")

with tab2:
    ref = st.file_uploader("Upload voix (6s) - I kan", type=["wav","mp3","m4a"])
    clone_txt = st.text_area("Texte Bamanankan", "I ni ce, n bɛ Bamanankan fɔ ka ɲɛ", height=100)
    if st.button("Cloner 🎤"):
        if ref is None:
            st.warning("Upload fɔlɔ!")
        else:
            out,clean = tts_long(clone_txt, "/tmp/clone.wav")
            st.write(clean); st.audio(out)
            st.info("MMS da ye Bamanankan tiɲɛ ye. XTTS tiɲɛ cloning bɛ Hugging Face Space de kan (Streamlit memory ka dɔgɔ).")
