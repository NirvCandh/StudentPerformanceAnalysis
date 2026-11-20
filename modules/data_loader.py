"""Pemuatan data, konstanta global, dan helper kecil yang dipakai semua modul."""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "StudentPerformanceFactors.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

TARGET = "Exam_Score"
CLASS_ORDER = ["Low Performance", "Medium Performance", "High Performance"]
CLASS_COLORS = {
    "Low Performance": "#EF553B",
    "Medium Performance": "#FECB52",
    "High Performance": "#00CC96",
}


def ensure_output_dirs() -> None:
    for d in (FIG_DIR, TABLE_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner="Membaca dataset...")
def load_data() -> pd.DataFrame:
    """Baca dataset langsung dari folder proyek (tanpa fitur upload)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"File '{DATA_PATH.name}' tidak ditemukan di folder proyek: {DATA_PATH.parent}"
        )
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ada di dataset.")
    return df


def split_columns(df: pd.DataFrame, target: str = TARGET):
    """Kembalikan (kolom numerik prediktor, kolom kategorikal prediktor); target dikeluarkan."""
    feats = df.drop(columns=[target], errors="ignore")
    num_cols = feats.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feats.columns if c not in num_cols]
    return num_cols, cat_cols


def info_box(text: str) -> None:
    """Kotak interpretasi singkat di bawah visualisasi."""
    st.info(f"**Interpretasi:** {text}")


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    ensure_output_dirs()
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index)


def save_figure(fig, name: str) -> None:
    """Simpan figur Plotly sebagai HTML interaktif (tidak butuh kaleido)."""
    ensure_output_dirs()
    fig.write_html(FIG_DIR / f"{name}.html", include_plotlyjs="cdn")


def show_fig(fig, name: str) -> None:
    """Tampilkan figur Plotly dan simpan ke outputs/figures."""
    st.plotly_chart(fig, key=f"plt_{name}")
    try:
        save_figure(fig, name)
    except Exception:  # penyimpanan hanya tambahan, jangan ganggu UI
        pass
