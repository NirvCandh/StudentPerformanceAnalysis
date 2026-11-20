"""Menu C - Classification: pembentukan kelas performa dari Exam_Score."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import CLASS_COLORS, CLASS_ORDER, TARGET, info_box, save_table, show_fig


def default_thresholds(scores: pd.Series) -> tuple[float, float]:
    """Default threshold = persentil ke-33,3 dan 66,7 dari data bersih (kelas relatif seimbang)."""
    q1, q2 = scores.quantile([1 / 3, 2 / 3])
    return round(float(q1), 1), round(float(q2), 1)


def assign_classes(scores: pd.Series, low_max: float, high_min: float) -> pd.Series:
    """Low: skor < low_max | Medium: low_max <= skor < high_min | High: skor >= high_min."""
    labels = np.select(
        [scores < low_max, scores < high_min],
        [CLASS_ORDER[0], CLASS_ORDER[1]],
        default=CLASS_ORDER[2],
    )
    return pd.Series(labels, index=scores.index, name="Performance_Class")


def get_thresholds() -> tuple[float, float]:
    """Threshold aktif; bila belum diatur, hitung default dari data bersih."""
    if "cfg_thresholds" not in st.session_state:
        st.session_state["cfg_thresholds"] = default_thresholds(st.session_state["clean_df"][TARGET])
    return st.session_state["cfg_thresholds"]


def class_table(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    return pd.DataFrame(
        {"Kelas": counts.index, "Jumlah siswa": counts.values, "Persentase (%)": (counts.values / counts.sum() * 100).round(2)}
    )


def render() -> None:
    st.header("🏷️ Classification - Pembentukan Kelas Performa")

    if "clean_df" not in st.session_state:
        st.warning("Jalankan **Data Preprocessing** terlebih dahulu agar kelas dibentuk dari data yang sudah bersih.")
        return

    clean = st.session_state["clean_df"]
    scores = clean[TARGET]
    smin, smax = float(scores.min()), float(scores.max())
    d_low, d_high = default_thresholds(scores)
    cur_low, cur_high = get_thresholds()

    st.markdown(
        f"Kelas dibentuk dari `{TARGET}`:\n"
        "- **Low Performance**: skor < batas bawah\n"
        "- **Medium Performance**: batas bawah <= skor < batas atas\n"
        "- **High Performance**: skor >= batas atas"
    )

    with st.container(border=True):
        st.subheader("Pengaturan threshold")
        c1, c2, c3 = st.columns([2, 2, 1])
        low_max = c1.number_input("Batas bawah (Low < nilai ini)", min_value=smin, max_value=smax,
                                  value=float(np.clip(cur_low, smin, smax)), step=0.5, key="in_low")
        high_min = c2.number_input("Batas atas (High >= nilai ini)", min_value=smin, max_value=smax,
                                   value=float(np.clip(cur_high, smin, smax)), step=0.5, key="in_high")
        c3.write("")
        c3.write("")
        if c3.button("Reset ke persentil 33/67"):
            st.session_state["cfg_thresholds"] = (d_low, d_high)
            st.session_state.pop("in_low", None)
            st.session_state.pop("in_high", None)
            st.rerun()
        st.caption(f"Default: persentil ke-33,3 dan ke-66,7 dari {TARGET} data bersih = {d_low} dan {d_high}.")

    if low_max >= high_min:
        st.error("Batas bawah harus lebih kecil daripada batas atas.")
        return

    labels = assign_classes(scores, low_max, high_min)
    counts = labels.value_counts()
    if any(counts.get(c, 0) < 2 for c in CLASS_ORDER):
        st.error("Ada kelas dengan kurang dari 2 siswa. Ubah threshold agar setiap kelas punya data yang cukup untuk stratified split.")
        return

    # Simpan threshold & label yang aktif
    st.session_state["cfg_thresholds"] = (low_max, high_min)
    st.session_state["class_labels"] = labels

    st.subheader("Dokumentasi threshold yang digunakan")
    doc = pd.DataFrame(
        {
            "Kelas": CLASS_ORDER,
            "Aturan": [f"{TARGET} < {low_max}", f"{low_max} <= {TARGET} < {high_min}", f"{TARGET} >= {high_min}"],
        }
    )
    st.dataframe(doc, hide_index=True)

    tbl = class_table(labels)
    save_table(tbl, "class_distribution", index=False)
    a, b = st.columns([1, 2])
    with a:
        st.subheader("Distribusi kelas")
        st.dataframe(tbl, hide_index=True)
    with b:
        fig = px.bar(tbl, x="Kelas", y="Jumlah siswa", color="Kelas", text="Jumlah siswa",
                     color_discrete_map=CLASS_COLORS, category_orders={"Kelas": CLASS_ORDER},
                     title="Distribusi Jumlah Siswa per Kelas Performa")
        fig.update_layout(xaxis_title="Kelas performa", yaxis_title="Jumlah siswa", legend_title="Kelas")
        show_fig(fig, "class_distribution_bar")

    biggest, smallest = tbl.loc[tbl["Jumlah siswa"].idxmax()], tbl.loc[tbl["Jumlah siswa"].idxmin()]
    ratio = biggest["Jumlah siswa"] / smallest["Jumlah siswa"]
    info_box(
        f"Kelas terbesar: {biggest['Kelas']} ({biggest['Persentase (%)']}%), terkecil: {smallest['Kelas']} ({smallest['Persentase (%)']}%). "
        + ("Distribusi relatif seimbang." if ratio < 1.5 else f"Kelas terbesar {ratio:.1f}x lebih banyak dari yang terkecil, sehingga ketidakseimbangan kelas perlu diperhatikan saat menilai model.")
    )

    # Histogram dengan garis threshold
    fig = px.histogram(clean, x=TARGET, nbins=40, title=f"Distribusi {TARGET} dan Batas Kelas",
                       labels={TARGET: "Exam Score"})
    fig.add_vline(x=low_max, line_dash="dash", line_color="red", annotation_text=f"Low < {low_max}")
    fig.add_vline(x=high_min, line_dash="dash", line_color="green", annotation_text=f"High >= {high_min}")
    fig.update_layout(xaxis_title="Exam Score", yaxis_title="Jumlah siswa")
    show_fig(fig, "class_threshold_hist")
    info_box("Garis putus-putus menunjukkan batas antarkelas pada distribusi nilai.")

    # Train vs Test
    st.subheader("Perbandingan distribusi kelas: data training vs testing")
    from modules.preprocessing import split_data  # import lokal menghindari circular import

    test_size = st.session_state.get("cfg_test_size", 0.2)
    seed = st.session_state.get("cfg_random_state", 42)
    try:
        _, _, y_tr, y_te = split_data(clean, labels, test_size, seed)
    except ValueError as e:
        st.error(f"Gagal melakukan split: {e}")
        return
    rows = []
    for name, y in (("Training", y_tr), ("Testing", y_te)):
        t = class_table(y)
        t["Subset"] = name
        rows.append(t)
    comp = pd.concat(rows, ignore_index=True)
    fig = px.bar(comp, x="Kelas", y="Persentase (%)", color="Subset", barmode="group", text="Persentase (%)",
                 category_orders={"Kelas": CLASS_ORDER}, title="Persentase Kelas pada Data Training vs Testing")
    fig.update_layout(xaxis_title="Kelas performa", yaxis_title="Persentase (%)", legend_title="Subset")
    show_fig(fig, "class_train_test_comparison")
    st.dataframe(comp.pivot(index="Kelas", columns="Subset", values="Jumlah siswa").reindex(CLASS_ORDER))
    info_box(
        f"Split stratified (test size {test_size:.0%}, random state {seed}) menjaga proporsi kelas pada training dan testing tetap hampir sama. "
        "Pengaturan test size dan random state dapat diubah di menu Modelling."
    )
