"""Menu A - Overview: gambaran umum dataset asli (belum diproses)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from modules.data_loader import TARGET, info_box, show_fig, split_columns


def _skew_text(skew: float) -> str:
    if abs(skew) < 0.5:
        return "cukup simetris"
    return "condong ke kanan (ekor panjang di nilai tinggi)" if skew > 0 else "condong ke kiri (ekor panjang di nilai rendah)"


def render(df: pd.DataFrame) -> None:
    st.header("📊 Overview Dataset")
    st.caption("Seluruh tampilan di menu ini memakai data asli apa adanya (sebelum preprocessing).")

    num_feats, cat_cols = split_columns(df)
    num_all = num_feats + [TARGET]
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())

    c = st.columns(6)
    c[0].metric("Jumlah baris", f"{df.shape[0]:,}")
    c[1].metric("Jumlah kolom", df.shape[1])
    c[2].metric("Atribut numerik", len(num_all))
    c[3].metric("Atribut kategorikal", len(cat_cols))
    c[4].metric("Missing values", f"{n_missing:,}")
    c[5].metric("Data duplikat", f"{n_dup:,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Ringkasan Data", f"Distribusi {TARGET}", "Variabel Numerik", "Variabel Kategorikal", "Korelasi"]
    )

    # ---- Ringkasan ----
    with tab1:
        st.subheader("Preview dataset")
        n = st.slider("Jumlah baris ditampilkan", 5, 50, 10, key="ov_preview_n")
        st.dataframe(df.head(n))
        left, right = st.columns(2)
        with left:
            st.subheader("Tipe data & missing values")
            info = pd.DataFrame(
                {
                    "Tipe data": df.dtypes.astype(str),
                    "Non-null": df.notna().sum(),
                    "Missing": df.isna().sum(),
                    "Missing (%)": (df.isna().mean() * 100).round(2),
                    "Nilai unik": df.nunique(),
                }
            )
            st.dataframe(info)
        with right:
            st.subheader("Statistik deskriptif (numerik)")
            st.dataframe(df[num_all].describe().T.round(3))
        miss = info[info["Missing"] > 0]
        if len(miss):
            info_box(
                "Kolom dengan missing values: "
                + ", ".join(f"{i} ({int(r['Missing'])} sel, {r['Missing (%)']}%)" for i, r in miss.iterrows())
                + ". Penanganannya dilakukan di menu Data Preprocessing lewat imputasi, bukan dengan menghapus baris."
            )
        else:
            info_box("Tidak ditemukan missing values pada dataset.")

    # ---- Exam_Score ----
    with tab2:
        s = df[TARGET].dropna()
        bins = st.slider("Jumlah bin histogram", 10, 80, 30, key="ov_bins")
        fig = px.histogram(
            df, x=TARGET, nbins=bins, marginal="box",
            title=f"Histogram Distribusi {TARGET}", labels={TARGET: "Exam Score", "count": "Jumlah siswa"},
        )
        fig.update_layout(yaxis_title="Jumlah siswa", bargap=0.05)
        show_fig(fig, "overview_hist_exam_score")
        info_box(
            f"Rata-rata {s.mean():.2f}, median {s.median():.2f}, simpangan baku {s.std():.2f}, "
            f"rentang {s.min():.0f}-{s.max():.0f}. Distribusi {_skew_text(s.skew())} (skewness {s.skew():.2f}). "
            "Titik di luar kotak boxplot adalah kandidat outlier yang diperiksa lebih lanjut di Data Preprocessing."
        )

        edges = np.arange(np.floor(s.min() / 5) * 5, np.ceil((s.max() + 1) / 5) * 5 + 5, 5)
        binned = pd.cut(s, bins=edges, right=False).value_counts().sort_index()
        bdf = pd.DataFrame({"Rentang nilai": binned.index.astype(str), "Jumlah siswa": binned.values})
        fig = px.bar(bdf, x="Rentang nilai", y="Jumlah siswa", text="Jumlah siswa",
                     title=f"Bar Chart {TARGET} per Rentang 5 Poin")
        fig.update_layout(xaxis_title="Rentang Exam Score", yaxis_title="Jumlah siswa")
        show_fig(fig, "overview_bar_exam_score_bins")
        top = bdf.loc[bdf["Jumlah siswa"].idxmax()]
        info_box(f"Rentang nilai paling banyak adalah {top['Rentang nilai']} dengan {int(top['Jumlah siswa'])} siswa.")

    # ---- Numerik ----
    with tab3:
        col = st.selectbox("Pilih variabel numerik", num_all, key="ov_num_col")
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram {col}")
            fig.update_layout(xaxis_title=col, yaxis_title="Jumlah siswa")
            show_fig(fig, f"overview_hist_{col}")
        with b:
            fig = px.box(df, y=col, points="outliers", title=f"Boxplot {col}")
            fig.update_layout(yaxis_title=col)
            show_fig(fig, f"overview_box_{col}")
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        info_box(
            f"{col} memiliki median {df[col].median():.2f} dan {n_out} titik di luar batas IQR (1,5 x IQR). "
            "Titik tersebut belum tentu salah; hanya nilai yang melanggar aturan validitas yang ditangani di preprocessing."
        )
        st.subheader("Boxplot semua variabel numerik (distandarkan agar sebanding)")
        z = (df[num_all] - df[num_all].mean()) / df[num_all].std()
        fig = px.box(z.melt(var_name="Variabel", value_name="Z-score"), x="Variabel", y="Z-score",
                     title="Boxplot Variabel Numerik (Z-score)")
        fig.update_layout(xaxis_title="Variabel", yaxis_title="Z-score")
        show_fig(fig, "overview_box_all_numeric")

    # ---- Kategorikal ----
    with tab4:
        if not cat_cols:
            st.warning("Tidak ada variabel kategorikal.")
        else:
            col = st.selectbox("Pilih variabel kategorikal", cat_cols, key="ov_cat_col")
            counts = df[col].fillna("(Missing)").value_counts().reset_index()
            counts.columns = [col, "Jumlah"]
            a, b = st.columns(2)
            with a:
                fig = px.bar(counts, x=col, y="Jumlah", color=col, text="Jumlah", title=f"Distribusi {col}")
                fig.update_layout(xaxis_title=col, yaxis_title="Jumlah siswa", legend_title=col)
                show_fig(fig, f"overview_bar_{col}")
            with b:
                fig = px.box(df, x=col, y=TARGET, color=col, title=f"{TARGET} berdasarkan {col}")
                fig.update_layout(xaxis_title=col, yaxis_title="Exam Score", legend_title=col)
                show_fig(fig, f"overview_box_score_by_{col}")
            top = counts.iloc[0]
            means = df.groupby(col)[TARGET].mean().sort_values()
            info_box(
                f"Kategori terbanyak pada {col} adalah '{top[col]}' ({int(top['Jumlah'])} siswa). "
                f"Rata-rata {TARGET} terendah pada kategori '{means.index[0]}' ({means.iloc[0]:.2f}) dan "
                f"tertinggi pada '{means.index[-1]}' ({means.iloc[-1]:.2f}). Ini gambaran deskriptif, bukan bukti sebab-akibat."
            )

    # ---- Korelasi ----
    with tab5:
        corr = df[num_all].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
        ax.set_title("Heatmap Korelasi Variabel Numerik (Pearson)")
        ax.set_xlabel("Variabel")
        ax.set_ylabel("Variabel")
        st.pyplot(fig)
        plt.close(fig)
        tc = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        if len(tc):
            info_box(
                f"Korelasi terkuat dengan {TARGET}: {tc.index[0]} ({tc.iloc[0]:+.2f})"
                + (f", lalu {tc.index[1]} ({tc.iloc[1]:+.2f})" if len(tc) > 1 else "")
                + ". Korelasi hanya mengukur hubungan linear antarvariabel numerik dan bukan bukti sebab-akibat."
            )
