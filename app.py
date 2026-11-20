"""Dashboard Analisis Student Performance (Streamlit).

Jalankan dengan:  streamlit run app.py
"""
import streamlit as st

from modules import classification, evaluation, modelling, overview, preprocessing
from modules.data_loader import DATA_PATH, TARGET, ensure_output_dirs, load_data

st.set_page_config(page_title="Student Performance Analysis", page_icon="🎓", layout="wide")
ensure_output_dirs()

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 0.8rem 1rem; border-radius: 0.6rem;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data()
except (FileNotFoundError, ValueError) as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

MENUS = {
    "📊 Overview": lambda: overview.render(df),
    "🧹 Data Preprocessing": lambda: preprocessing.render(df),
    "🏷️ Classification": classification.render,
    "🤖 Modelling": lambda: modelling.render(df),
    "📈 Evaluation": evaluation.render,
}

with st.sidebar:
    st.title("🎓 Student Performance")
    st.caption(f"Dataset: {DATA_PATH.name}  \nTarget: `{TARGET}`")
    page = st.radio("Navigasi", list(MENUS), label_visibility="collapsed")
    st.divider()
    st.markdown("**Status proses**")
    has_clean = "clean_df" in st.session_state
    st.write(("✅" if has_clean else "⬜") + " Preprocessing")
    if has_clean:
        lo, hi = classification.get_thresholds()
        st.write(f"✅ Classification (threshold {lo} / {hi})")
    else:
        st.write("⬜ Classification")
    st.write(("✅" if "results" in st.session_state else "⬜") + " Modelling")
    if st.button("Reset hasil proses"):
        for k in ("clean_df", "prep_report", "prep_snapshot", "class_labels", "cfg_thresholds", "in_low", "in_high", "results"):
            st.session_state.pop(k, None)
        st.rerun()

st.title("Analisis Faktor Performa Akademik Siswa")
try:
    MENUS[page]()
except Exception as e:  # noqa: BLE001
    st.error(f"Terjadi kesalahan pada menu '{page}': {e}")
    st.exception(e)
