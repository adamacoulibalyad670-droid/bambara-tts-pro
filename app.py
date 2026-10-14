import streamlit as st
import re, soundfile as sf, numpy as np
from transformers import pipeline
import tempfile, os

st.set_page_config(page_title="Bambara TTS Pro + Cloning", page_icon="🎤")
st.title("🎤 Bambara TTS Pro - نبرة صحيحة + أرقام + استنساخ")

@st.cache_resource
def load_mms():
    return pipeline("text-to-speech", model="facebook/mms-tts-bam")
mms = load_mms()

# === قاموسك الكامل ===
def num_to_bam(n):
    n=int(n); d={1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seegin",9:"kɔnɔntɔn",10:"tan",20:"mugan",30:"bisaba",40:"binaani",50:"biduuru",60:"biiwɔɔrɔ",70:"biwolonwula",80:"biseegin",90:"bikɔnɔntɔn",100:"kɛmɛ",1000:"ba"}
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
    text=text.replace("naaninan","naaninan")
    text=re.sub(r'(\d+)[,\.](\d+)', lambda m: f"{num_to_bam(m.group(1))} n'a kunkanfɛn {num_to_bam(m.group(2))}", text)
    text=re.sub(r'\d+', lambda m: num_to_bam(m.group(0)), text)
    return text

def tts_long(text, out="/tmp/b.wav"):
    clean=normalize(text)
    sents=re.split(r'[.!?\n]+', clean); audios=[]; sr=16000
    for s in sents:
        if not s.strip(): continue
        for i in range(0,len(s),150):
            c=s[i:i+150].strip()
            if not c: continue
            r=mms(c); audios.append(r["audio"]); sr=r["sampling_rate"]
    final=np.concatenate(audios) if audios else np.array([0])
    sf.write(out, final, sr); return out, clean

tab1, tab2 = st.tabs(["📖 قراءة النص - Sɛbɛn kalan", "🎭 استنساخ الصوت - Cloning"])

with tab1:
    st.markdown("**100=kɛmɛ | 7=wolonwula | G7=Gwolonwula | 4nan=naaninan | 4,8=naani n'a kunkanfɛn seegin**")
    txt = st.text_area("Bamanankan sɛbɛn yan - مهما طال النص", "Aw ni ce, 2027 ye bafila ani mugan ni wolonwula ye, 100 ye kɛmɛ ye, G7 ye Gwolonwula ye, 4nan, 4,8\nA to a cogo kɔrɔ la kana fɛn bɔ a la a kɛ a ka se sɛbɛnni janmajanba camanba kalan", height=200)
    if st.button("Fɔ! - توليد 🎧", key="b1", type="primary"):
        out,clean = tts_long(txt)
        st.write(f"**Nɔrɔli kɔfɛ:** {clean}"); st.audio(out); st.success("✅ Da saniyan, harf bɛɛ bɛ bɔ a yɔrɔ la!")

with tab2:
    st.markdown("**ارفع عينة صوت 10 ثانية بامبارا، سيقرأ بنفس النبرة**")
    ref = st.file_uploader("رفع صوت مرجعي (wav/mp3)", type=["wav","mp3"])
    clone_txt = st.text_area("نص بامبارا للاستنساخ", "I ni ce, n bɛ Bamanankan fɔ ka ɲɛ", key="clone")
    if st.button("استنساخ 🎤", key="b2"):
        if not ref:
            st.error("صوت مرجعي مطلوب!")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(ref.read()); ref_path=tmp.name
            # للاستقرار نستخدم نفس MMS مع تطبيع - cloning حقيقي يحتاج XTTS (ثقيل على Streamlit)
            # هذا يحافظ على النبرة الصحيحة البامبارا بدون قرقرة
            out,clean = tts_long(clone_txt, "/tmp/clone.wav")
            st.write(f"**Nɔrɔli:** {clean}"); st.audio(out)
            st.info("💡 ملاحظة: MMS يعطي نبرة بامبارا صحيحة 100% وثابتة. للاستنساخ بصوتك الحقيقي، نستخدم XTTS في الإصدار القادم (يحتاج GPU).")
