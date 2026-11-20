"""Menu E - Evaluation: evaluasi mendalam model hasil Modelling."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import CLASS_ORDER, info_box, save_table, show_fig
from modules.modelling import METRIC_OPTIONS, best_model_name, results_table, stale_warning


def _quality(acc: float) -> str:
    if acc >= 0.9:
        return "sangat baik"
    if acc >= 0.8:
        return "baik"
    if acc >= 0.7:
        return "cukup"
    if acc >= 0.6:
        return "sedang"
    return "rendah"


def render() -> None:
    st.header("📈 Evaluation")
    results = st.session_state.get("results")
    if results is None:
        st.warning("Belum ada hasil. Latih model di menu **Modelling** terlebih dahulu.")
        return

    stale_warning(results)
    tbl = results_table(results)
    names = list(results["models"])
    metric_label = st.selectbox("Metrik penentu model terbaik", list(METRIC_OPTIONS), key="eval_metric")
    best = best_model_name(results, METRIC_OPTIONS[metric_label])
    name = st.selectbox("Model yang dievaluasi", names, index=names.index(best), key="eval_model",
                        help="Default: model terbaik berdasarkan metrik di atas.")
    if name == best:
        st.caption("Ini adalah model terbaik berdasarkan metrik terpilih.")

    r = results["models"][name]
    m = r["metrics"]
    y_test = results["y_test"]

    st.subheader(f"Ringkasan metrik - {name}")
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['Accuracy']:.4f}")
    c[1].metric("Precision (weighted)", f"{m['Precision (weighted)']:.4f}")
    c[2].metric("Recall (weighted)", f"{m['Recall (weighted)']:.4f}")
    c[3].metric("F1-Score (weighted)", f"{m['F1 (weighted)']:.4f}")
    baseline = y_test.value_counts(normalize=True).max()
    gap = r["train_accuracy"] - m["Accuracy"]
    info_box(
        f"Akurasi {m['Accuracy']:.1%} tergolong {_quality(m['Accuracy'])} (patokan umum, bukan standar baku). "
        f"Sebagai pembanding, selalu menebak kelas mayoritas pada data testing menghasilkan akurasi {baseline:.1%}; "
        + ("model lebih baik dari tebakan tersebut" if m["Accuracy"] > baseline else "model belum lebih baik dari tebakan kelas mayoritas")
        + f". Precision menunjukkan seberapa tepat prediksi suatu kelas, recall seberapa banyak siswa suatu kelas yang berhasil ditemukan, F1 adalah keseimbangan keduanya. "
        + (f"Akurasi training ({r['train_accuracy']:.1%}) jauh di atas testing ({m['Accuracy']:.1%}), indikasi overfitting." if gap > 0.1
           else f"Akurasi training ({r['train_accuracy']:.1%}) dan testing ({m['Accuracy']:.1%}) berdekatan, tidak ada tanda overfitting yang kuat.")
    )

    # ---- Confusion matrix ----
    st.subheader("Confusion Matrix")
    norm = st.radio("Tampilan", ["Jumlah", "Dinormalisasi per kelas aktual (%)"], horizontal=True)
    cm = r["cm"]
    if norm == "Jumlah":
        z, text, fmt = cm, cm.astype(str), None
    else:
        z = cm / cm.sum(axis=1, keepdims=True) * 100
        text = np.vectorize(lambda v: f"{v:.1f}%")(z)
    fig = go.Figure(go.Heatmap(z=z, x=CLASS_ORDER, y=CLASS_ORDER, text=text, texttemplate="%{text}",
                               colorscale="Blues", colorbar=dict(title="Nilai")))
    fig.update_layout(title=f"Confusion Matrix - {name}", xaxis_title="Kelas prediksi", yaxis_title="Kelas aktual",
                      yaxis_autorange="reversed")
    show_fig(fig, f"eval_confusion_{name.replace(' ', '_')}")
    row_recall = cm.diagonal() / cm.sum(axis=1)
    worst = int(np.argmin(row_recall))
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    msg = (f"Diagonal utama adalah prediksi benar. Kelas yang paling sulit dikenali: {CLASS_ORDER[worst]} (recall {row_recall[worst]:.1%}). ")
    if off[i, j] > 0:
        msg += f"Kesalahan terbanyak: siswa {CLASS_ORDER[i]} diprediksi sebagai {CLASS_ORDER[j]} ({int(off[i, j])} siswa)."
    info_box(msg)

    # ---- Classification report ----
    st.subheader("Classification Report")
    rep = pd.DataFrame(r["report"]).T
    rep = rep.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-Score", "support": "Support"})
    st.dataframe(rep.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "Support": "{:.0f}"}))
    save_table(rep, f"classification_report_{name.replace(' ', '_')}")
    per = rep.loc[CLASS_ORDER, ["Precision", "Recall", "F1-Score"]].reset_index(names="Kelas").melt(
        id_vars="Kelas", var_name="Metrik", value_name="Skor")
    fig = px.bar(per, x="Kelas", y="Skor", color="Metrik", barmode="group", category_orders={"Kelas": CLASS_ORDER},
                 title=f"Precision, Recall, F1-Score per Kelas - {name}")
    fig.update_layout(xaxis_title="Kelas performa", yaxis_title="Skor", yaxis_range=[0, 1], legend_title="Metrik")
    show_fig(fig, f"eval_per_class_{name.replace(' ', '_')}")
    f1c = rep.loc[CLASS_ORDER, "F1-Score"]
    info_box(f"F1 per kelas tertinggi: {f1c.idxmax()} ({f1c.max():.3f}); terendah: {f1c.idxmin()} ({f1c.min():.3f}). "
             "'macro avg' merata-ratakan kelas secara setara, sedangkan 'weighted avg' memperhitungkan jumlah siswa tiap kelas.")

    # ---- Perbandingan antar model ----
    st.subheader("Perbandingan seluruh algoritma")
    fig = px.bar(tbl.reset_index(), x="Algoritma", y="Accuracy", color="Algoritma", text=tbl["Accuracy"].round(4).values,
                 title="Perbandingan Accuracy Antaralgoritma")
    fig.update_layout(xaxis_title="Algoritma", yaxis_title="Accuracy", yaxis_range=[0, 1], showlegend=False)
    show_fig(fig, "eval_accuracy_comparison")

    cols = ["Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
    long = tbl[cols].reset_index().melt(id_vars="Algoritma", var_name="Metrik", value_name="Skor")
    fig = px.bar(long, x="Algoritma", y="Skor", color="Metrik", barmode="group",
                 title="Perbandingan Precision, Recall, dan F1-Score (weighted)")
    fig.update_layout(xaxis_title="Algoritma", yaxis_title="Skor", yaxis_range=[0, 1], legend_title="Metrik")
    show_fig(fig, "eval_prf_comparison")

    radar_cols = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "F1 (macro)"]
    fig = go.Figure()
    for algo in tbl.index:
        v = tbl.loc[algo, radar_cols].tolist()
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=radar_cols + radar_cols[:1], name=algo, fill="toself", opacity=0.45))
    lo = max(0.0, float(tbl[radar_cols].min().min()) - 0.05)
    fig.update_layout(title="Grafik Performa Model (Radar)", polar=dict(radialaxis=dict(range=[lo, 1.0])), legend_title="Algoritma")
    show_fig(fig, "eval_radar")
    top = tbl["Accuracy"].idxmax()
    info_box(f"Sumbu radar dimulai dari {lo:.2f} agar perbedaan antarmodel terlihat; bandingkan nilai pada tabel untuk selisih sebenarnya. "
             f"Akurasi tertinggi: {top} ({tbl.loc[top, 'Accuracy']:.4f}).")

    with st.expander("Tabel lengkap seluruh metrik"):
        st.dataframe(tbl.style.format("{:.4f}"))
