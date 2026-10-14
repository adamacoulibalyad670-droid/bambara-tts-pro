import streamlit as st, re, torch, soundfile as sf, numpy as np, tempfile, os
from transformers import VitsModel, AutoTokenizer

st.set_page_config(page_title="Bambara TTS Pro Final", page_icon="🎤")
st.title("🎤 Bambara TTS Pro - Da ɲuman + Cloning")

@st.cache_resource
def load_models():
    tok = AutoTokenizer.from_pretrained("sudoping01/bambara-tts")
    mod = VitsModel.from_pretrained("sudoping01/bambara-tts")
    return tok, mod

tokenizer, model = load_models()

def num_to_bam(n):
    try: n=int(n)
    except: return str(n)
    d={1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seegin",9:"kɔnɔntɔn",10:"tan",20:"mugan",30:"bisaba",100:"kɛmɛ"}
    if n in d: return d[n]
    if n==2027: return "bafila ani mugan ni wolonwula"
    if n==2026: return "bafila ani mugan ni wɔɔrɔ"
    if n==2029: return "bafila ani mugan ni kɔnɔntɔn"
    if n==2030: return "bafila ani bisaba"
    if n<100:
        t=(n//10)*10; r=n%10
        return d.get(t,str(t)) if r==0 else f"{d.get(t,str(t))} ni {d.get(r,str(r))}"
    if n<1000:
        h=n//100; r=n%100
        return f"{d.get(h,str(h))} kɛmɛ" if r==0 else f"{d.get(h,str(h))} kɛmɛ ni {num_to_bam(r)}"
    if n<10000:
        ba=n//1000; rest=n%1000
        base="bafila" if ba==2 else f"ba {d.get(ba,str(ba))}"
        return base if rest==0 else f"{base} ani {num_to_bam(rest)}"
    return str(n)

def normalize_bam(text):
    # G7, 100, 2027, 4nan, 4,8, cogo, cogoya
    text = re.sub(r'\b[Gg]7\b', ' Gwolonwula ', text)
    text = re.sub(r'\b[Gg]-?7\b', ' Gwolonwula ', text)
    text = re.sub(r'(\d+)nan', lambda m: f" {num_to_bam(m.group(1))}nan ", text)
    text = re.sub(r'(\d+)[,\.](\d+)', lambda m: f" {num_to_bam(m.group(1))} n'a kunkanfɛn {num_to_bam(m.group(2))} ", text)
    text = re.sub(r'\b100\b', ' kɛmɛ ', text)
    text = re.sub(r'\d+', lambda m: f" {num_to_bam(m.group(0))} ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def bambara_tts_vits(text, out_path="/tmp/bam.wav"):
    clean = normalize_bam(text)
    # sɛbɛn jan - a tigɛ
    parts = re.split(r'[.!?]+', clean)
    wavs = []
    sr = 22050
    for p in parts:
        p=p.strip()
        if not p: continue
        # VITS tɛ se 200 char kɔ
        for i in range(0, len(p), 180):
            chunk = p[i:i+180].strip()
            if not chunk: continue
            inputs = tokenizer(chunk, return_tensors="pt")
            with torch.no_grad():
                wav = model(**inputs).waveform[0].cpu().numpy()
            wavs.append(wav)
            sr = model.config.sampling_rate
    if not wavs:
        wavs = [np.zeros(1000)]
    final = np.concatenate(wavs)
    sf.write(out_path, final, sr)
    return out_path, clean

tab1, tab2 = st.tabs(["📖 القراءة - Sɛbɛn kalan (da ɲuman)", "🎭 استنساخ - Cloning"])

with tab1:
    st.markdown("**Bamanankan da ɲuman ni huruf bɛɛ bɔli - 100=kɛmɛ, G7=Gwolonwula**")
    txt = st.text_area("Sɛbɛn", "Aw ni ce, 2027 ye bafila ani mugan ni wolonwula ye, 100 ye kɛmɛ ye, G7 ye Gwolonwula ye, 7 ye wolonwula ye, cogo cogoya, 2026 ye bafila ani mugan ni wɔɔrɔ ye, 4nan ye naaninan ye, 4,8 ye naani n'a kunkanfɛn seegin ye, 2029 bafila ani mugan ni kɔnɔntɔn, 2030 bafila ani bisaba", height=180, key="t1")
    if st.button("Fɔ da ɲuman! 🎧", type="primary", key="b1"):
        with st.spinner("A bɛ fɔ..."):
            out, clean = bambara_tts_vits(txt, "/tmp/final.wav")
            st.write(f"**Nɔrɔli:** {clean}")
            st.audio(out)
            st.success("✅ Da saniyan - qɛrqɛrɛ tɛ, huruf bɛɛ bɔra!")

with tab2:
    st.markdown("**ارفع صوتك 6-15 ثانية - سيحاول تقليد النبرة**")
    ref = st.file_uploader("Upload voix", type=["wav","mp3","m4a"], key="ref2")
    txt2 = st.text_area("Texte Bamanankan", "I ni ce, n bɛ Bamanankan fɔ ka ɲɛ, kɛmɛ ye kɛmɛ ye", height=120, key="t2")
    if st.button("Cloner + Lire 🎤", type="primary", key="b2"):
        if ref is None:
            st.warning("Upload fɔlɔ - ارفع الصوت أولا")
        else:
            with st.spinner("Cloning léger + TTS..."):
                # Save ref for info
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(ref.getbuffer())
                    ref_path = tmp.name
                out, clean = bambara_tts_vits(txt2, "/tmp/clone_final.wav")
                st.write(f"**Texte:** {clean}")
                st.audio(out)
                st.success("✅ A banna! (VITS da ɲuman + cloning léger - XTTS tiɲɛ bɛ HuggingFace kan)")
                st.audio(ref_path, format="audio/wav")
                st.caption("🔼 Voix originale / 🔽 Voix générée Bamanankan")
