import streamlit as st
import re, numpy as np, librosa, soundfile as sf
from transformers import pipeline
import torch

st.set_page_config(page_title="Bambara TTS + Cloning", page_icon="🎤")
st.title("🎤 Bambara TTS - Cloning Léger V2")
st.caption("100=kɛmɛ | G7=Gwolonwula | 2027=bafila ani mugan ni wolonwula")

@st.cache_resource
def load_pipe():
    return pipeline("text-to-speech", model="facebook/mms-tts-bam")

def num_to_bam(n):
    try: n=int(n)
    except: return str(n)
    d={1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seegin",9:"kɔnɔntɔn",10:"tan",20:"mugan",30:"bisaba",100:"kɛmɛ"}
    if n in d: return d[n]
    if n==2027: return "bafila ani mugan ni wolonwula"
    if n==2026: return "bafila ani mugan ni wɔɔrɔ"
    if n==2029: return "bafila ani mugan ni kɔnɔntɔn"
    if n==2030: return "bafila ani bisaba"
    if n<100: t=(n//10)*10; r=n%10; return d[t] if r==0 else f"{d[t]} ni {d[r]}"
    if n<1000: return f"{d[n//100]} kɛmɛ" if n%100==0 else f"{d[n//100]} kɛmɛ ni {num_to_bam(n%100)}"
    if n<10000: ba=n//1000; rest=n%1000; base="bafila" if ba==2 else f"ba {d[ba]}"; return base if rest==0 else f"{base} ani {num_to_bam(rest)}"
    return str(n)

def normalize(t):
    t=re.sub(r'\b[Gg]7\b',' Gwolonwula ',t)
    t=re.sub(r'(\d+)nan',lambda m:f" {num_to_bam(m.group(1))}nan ",t)
    t=re.sub(r'(\d+)[,\.](\d+)',lambda m:f" {num_to_bam(m.group(1))} n'a kunkanfɛn {num_to_bam(m.group(2))} ",t)
    t=re.sub(r'\b100\b',' kɛmɛ ',t)
    t=re.sub(r'\d+',lambda m:f" {num_to_bam(m.group(0))} ",t)
    t=re.sub(r'\s+',' ',t).strip()
    return t

# UI
ref = st.file_uploader("1️⃣ Upload i kan (muso/cɛ) WAV/MP3", type=["wav","mp3","m4a"])
text = st.text_area("2️⃣ Bamanankan + jatew", value="Aw ni ce, 2027 ye bafila ani mugan ni wolonwula ye, 100 ye kɛmɛ ye, G7 ye Gwolonwula ye, 4nan ye naaninan ye", height=120)

if st.button("🎧 Fɔ i kan na!", type="primary"):
    if not text.strip():
        st.warning("Texte sɛbɛn!")
    else:
        with st.spinner("A bɛ fɔ..."):
            clean = normalize(text)
            st.info(f"Nɔrɔli: {clean}")
            pipe = load_pipe()
            # TTS
            sents = re.split(r'[.!?]+', clean)
            audios=[]; sr=16000
            for s in sents:
                if not s.strip(): continue
                for i in range(0,len(s),140):
                    c=s[i:i+140].strip()
                    if not c: continue
                    r=pipe(c); audios.append(r["audio"]); sr=r["sampling_rate"]
            wav = np.concatenate(audios) if audios else np.zeros(1000)

            # Cloning V2
            if ref is not None:
                try:
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(ref.read()); tmp_path=tmp.name
                    y_ref, _ = librosa.load(tmp_path, sr=sr, duration=15)
                    os.unlink(tmp_path)
                    f0_ref = librosa.yin(y_ref, fmin=50, fmax=400)
                    f0_ref = f0_ref[f0_ref>50]
                    if len(f0_ref)>0:
                        mean_f0 = np.median(f0_ref)
                        n_steps = 12 * np.log2(mean_f0 / 130)
                        n_steps = np.clip(n_steps, -8, 8)
                        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=float(n_steps))
                        st.success(f"Cloning: pitch {n_steps:.1f} semitones - {'Muso' if mean_f0>160 else 'Cɛ'} kan!")
                except Exception as e:
                    st.error(f"Cloning error: {e}")

            sf.write("/tmp/out.wav", wav, sr)
            st.audio("/tmp/out.wav")
            st.download_button("📥 Download", open("/tmp/out.wav","rb"), "bambara_clone.wav")

st.markdown("---")
st.caption("Cloning léger: pitch bɛ yɛlɛma | Cloning tiɲɛ (XTTS) bɛ Colab/PRO dɔrɔn de la")
