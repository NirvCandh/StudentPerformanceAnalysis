"""Menu B - Data Preprocessing.

Prinsip: dataset asli tidak diubah (selalu bekerja pada salinan). Imputasi, scaling, dan
encoding dilakukan di dalam Pipeline/ColumnTransformer sehingga statistiknya hanya
dipelajari dari data training (mencegah data leakage).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from modules.classification import assign_classes, get_thresholds
from modules.data_loader import TARGET, info_box, save_table, show_fig, split_columns

# Aturan validitas berbasis domain (bukan statistik data): (minimum, maksimum); None = tanpa batas.
VALID_RANGES = {
    "Attendance": (0, 100),
    "Exam_Score": (0, 100),
    "Previous_Scores": (0, 100),
    "Hours_Studied": (0, 168),   # jam per minggu
    "Sleep_Hours": (0, 24),      # jam per hari
    "Tutoring_Sessions": (0, None),
    "Physical_Activity": (0, None),
}
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}


# ---------------------------------------------------------------- fungsi pemeriksaan
def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    t = pd.DataFrame({"Missing": m, "Missing (%)": (m / len(df) * 100).round(2), "Tipe": df.dtypes.astype(str)})
    return t[t["Missing"] > 0].sort_values("Missing", ascending=False)


def outlier_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"Variabel": c, "Batas bawah IQR": round(lo, 2), "Batas atas IQR": round(hi, 2),
                     "Jumlah outlier": n, "Outlier (%)": round(n / len(s) * 100, 2)})
    return pd.DataFrame(rows)


def invalid_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, (lo, hi) in VALID_RANGES.items():
        if c not in df.columns:
            continue
        s = df[c]
        below = int((s < lo).sum()) if lo is not None else 0
        above = int((s > hi).sum()) if hi is not None else 0
        rows.append({"Variabel": c, "Rentang valid": f"{'-inf' if lo is None else lo} s/d {'inf' if hi is None else hi}",
                     "Di bawah rentang": below, "Di atas rentang": above, "Total tidak valid": below + above})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fungsi pemrosesan
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan salinan data. Imputasi TIDAK dilakukan di sini (dilakukan di pipeline).

    1. Hapus baris duplikat.
    2. Baris dengan Exam_Score tidak valid dihapus (target tidak boleh diimputasi/dikarang).
    3. Nilai fitur yang melanggar aturan validitas diubah menjadi NaN agar diimputasi dari data training.
    4. Outlier statistik (IQR) dipertahankan.
    """
    out = df.copy()
    report = {"rows_raw": len(out)}
    report["duplicates_removed"] = int(out.duplicated().sum())
    out = out.drop_duplicates()

    lo, hi = VALID_RANGES[TARGET]
    bad_target = out[TARGET].notna() & ((out[TARGET] < lo) | (out[TARGET] > hi))
    report["invalid_target_rows_removed"] = int(bad_target.sum())
    out = out[~bad_target]
    report["missing_target_rows_removed"] = int(out[TARGET].isna().sum())
    out = out[out[TARGET].notna()]

    to_nan = {}
    for c, (lo, hi) in VALID_RANGES.items():
        if c == TARGET or c not in out.columns:
            continue
        bad = pd.Series(False, index=out.index)
        if lo is not None:
            bad |= out[c] < lo
        if hi is not None:
            bad |= out[c] > hi
        if bad.any():
            to_nan[c] = int(bad.sum())
            out.loc[bad, c] = np.nan
    report["invalid_features_set_to_nan"] = to_nan
    out = out.reset_index(drop=True)
    report["rows_clean"] = len(out)
    return out, report


def build_preprocessor(num_cols, cat_cols, num_strategy="median", scaler="StandardScaler") -> ColumnTransformer:
    """ColumnTransformer yang dapat dipakai ulang di Modelling. Belum di-fit."""
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy=num_strategy)), ("scaler", SCALERS[scaler]())])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, list(num_cols)), ("cat", cat_pipe, list(cat_cols))],
        sparse_threshold=0.0,
    )


def split_data(clean_df: pd.DataFrame, labels: pd.Series, test_size: float, random_state: int):
    """Pisahkan fitur & target, lalu stratified train-test split. Exam_Score TIDAK menjadi fitur."""
    X = clean_df.drop(columns=[TARGET])
    return train_test_split(X, labels, test_size=test_size, random_state=random_state, stratify=labels)


def get_prep_cfg() -> dict:
    return st.session_state.setdefault("cfg_prep", {"num_strategy": "median", "scaler": "StandardScaler"})


# ---------------------------------------------------------------- tampilan
def render(df: pd.DataFrame) -> None:
    st.header("🧹 Data Preprocessing")
    st.caption("Dataset asli tetap utuh; semua langkah bekerja pada salinan.")
    num_cols, cat_cols = split_columns(df)

    # ---- 1. Diagnosis ----
    st.subheader("1. Pemeriksaan data asli")
    mt, inv, out = missing_table(df), invalid_table(df), outlier_table(df, num_cols + [TARGET])
    c = st.columns(4)
    c[0].metric("Total missing values", int(df.isna().sum().sum()))
    c[1].metric("Kolom dengan missing", len(mt))
    c[2].metric("Data duplikat", int(df.duplicated().sum()))
    c[3].metric("Nilai tidak valid", int(inv["Total tidak valid"].sum()) if len(inv) else 0)

    t1, t2, t3 = st.tabs(["Missing values", "Nilai tidak valid", "Outlier (IQR)"])
    with t1:
        if len(mt):
            st.dataframe(mt)
            fig = px.bar(mt.reset_index(names="Kolom"), x="Kolom", y="Missing", text="Missing", title="Missing Values per Kolom (Data Asli)")
            fig.update_layout(xaxis_title="Kolom", yaxis_title="Jumlah missing")
            show_fig(fig, "prep_missing_raw")
            info_box("Missing values pada kolom kategorikal diimputasi dengan modus, pada kolom numerik dengan median/mean (pilihan di bawah). Baris tidak dihapus agar tidak membuang informasi.")
        else:
            st.success("Tidak ada missing values.")
    with t2:
        st.dataframe(inv, hide_index=True)
        info_box("Aturan memakai batas logis domain (mis. Attendance 0-100, Exam_Score 0-100, jam tidur 0-24). "
                 "Nilai fitur yang melanggar dijadikan missing lalu diimputasi; baris dengan Exam_Score tidak valid dihapus karena target tidak boleh dikarang.")
    with t3:
        st.dataframe(out, hide_index=True)
        col = st.selectbox("Boxplot variabel", num_cols + [TARGET], key="prep_out_col")
        fig = px.box(df, y=col, points="outliers", title=f"Boxplot {col} (Data Asli)")
        fig.update_layout(yaxis_title=col)
        show_fig(fig, f"prep_box_{col}")
        info_box("Outlier IQR hanya ditandai, tidak dibuang otomatis: nilai ekstrem bisa saja valid (mis. siswa yang belajar sangat lama). "
                 "Hanya nilai yang melanggar aturan validitas yang ditangani.")

    # ---- 2. Pengaturan & eksekusi ----
    st.subheader("2. Pengaturan & eksekusi")
    cfg = get_prep_cfg()
    a, b = st.columns(2)
    cfg["num_strategy"] = a.selectbox("Imputasi numerik", ["median", "mean"], index=["median", "mean"].index(cfg["num_strategy"]))
    cfg["scaler"] = b.selectbox("Scaling numerik", list(SCALERS), index=list(SCALERS).index(cfg["scaler"]))
    st.caption("Imputasi kategorikal: modus (most_frequent). Encoding: One-Hot Encoding. Semuanya di dalam ColumnTransformer yang di-fit hanya pada data training.")

    if st.button("▶️ Jalankan Preprocessing", type="primary"):
        try:
            clean, report = clean_data(df)
            st.session_state["clean_df"] = clean
            st.session_state["prep_report"] = report
            st.session_state["prep_snapshot"] = dict(cfg)
            for k in ("results", "class_labels", "cfg_thresholds", "in_low", "in_high"):  # hasil lama tidak lagi valid
                st.session_state.pop(k, None)
            save_table(clean.describe(include="all").T, "clean_data_summary")
            st.success("Preprocessing selesai.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Preprocessing gagal: {e}")
            return

    if "clean_df" not in st.session_state:
        st.info("Klik **Jalankan Preprocessing** untuk melihat hasil.")
        return

    _render_results(df, num_cols, cat_cols)


def _render_results(raw: pd.DataFrame, num_cols, cat_cols) -> None:
    clean, report = st.session_state["clean_df"], st.session_state["prep_report"]
    snap = st.session_state["prep_snapshot"]
    if snap != get_prep_cfg():
        st.warning("Pengaturan preprocessing diubah setelah proses terakhir. Klik **Jalankan Preprocessing** lagi agar berlaku.")

    st.subheader("3. Hasil pembersihan")
    c = st.columns(4)
    c[0].metric("Baris sebelum", f"{report['rows_raw']:,}")
    c[1].metric("Baris sesudah", f"{report['rows_clean']:,}", delta=report["rows_clean"] - report["rows_raw"])
    c[2].metric("Duplikat dihapus", report["duplicates_removed"])
    c[3].metric("Baris Exam_Score tidak valid dihapus", report["invalid_target_rows_removed"] + report["missing_target_rows_removed"])
    if report["invalid_features_set_to_nan"]:
        st.write("Nilai fitur tidak valid yang diubah menjadi missing (lalu diimputasi): "
                 + ", ".join(f"{k}: {v}" for k, v in report["invalid_features_set_to_nan"].items()))
    else:
        st.write("Tidak ada nilai fitur yang melanggar aturan validitas.")

    # Fit preprocessor pada training saja
    labels = assign_classes(clean[TARGET], *get_thresholds())
    test_size = st.session_state.get("cfg_test_size", 0.2)
    seed = st.session_state.get("cfg_random_state", 42)
    try:
        X_tr, X_te, y_tr, y_te = split_data(clean, labels, test_size, seed)
    except ValueError as e:
        st.error(f"Train-test split gagal: {e}")
        return
    pre = build_preprocessor(num_cols, cat_cols, snap["num_strategy"], snap["scaler"])
    Xt_tr = pre.fit_transform(X_tr)
    Xt_te = pre.transform(X_te)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    Xt_tr_df = pd.DataFrame(Xt_tr, columns=names)

    st.subheader("4. Pemisahan fitur-target & train-test split")
    c = st.columns(4)
    c[0].metric("Fitur (sebelum encoding)", X_tr.shape[1])
    c[1].metric("Fitur (setelah One-Hot)", Xt_tr.shape[1])
    c[2].metric("Data training", f"{len(X_tr):,}")
    c[3].metric("Data testing", f"{len(X_te):,}")
    st.caption(f"Split stratified berdasarkan kelas performa (test size {test_size:.0%}, random state {seed}); `{TARGET}` tidak ikut sebagai fitur. "
               "Pengaturan dapat diubah di menu Modelling.")
    st.write("Sampel data training setelah transformasi:")
    st.dataframe(Xt_tr_df.head(10).round(3))
    nan_left = int(np.isnan(Xt_tr).sum() + np.isnan(Xt_te).sum())
    info_box(f"Setelah pipeline, jumlah nilai kosong pada data training+testing = {nan_left}. "
             f"Kolom kategorikal diubah menjadi {Xt_tr.shape[1] - len(num_cols)} kolom biner; {len(num_cols)} kolom numerik diskalakan dengan {snap['scaler']}. "
             "Imputer, scaler, dan encoder hanya mempelajari data training sehingga tidak ada kebocoran dari data testing.")

    st.subheader("5. Perbandingan sebelum vs sesudah preprocessing")
    # (a) missing: dihitung nyata pada tiga tahap
    imputed = pd.concat([X_tr, X_te]).copy()
    imputed[num_cols] = pre.named_transformers_["num"].named_steps["imputer"].transform(imputed[num_cols])
    if cat_cols:
        imputed[cat_cols] = pre.named_transformers_["cat"].named_steps["imputer"].transform(imputed[cat_cols])
    stages = pd.DataFrame({
        "Data asli": raw.isna().sum(),
        "Setelah pembersihan": clean.isna().sum(),
        "Setelah imputasi": imputed.isna().sum().reindex(raw.columns).fillna(0).astype(int),
    })
    stages = stages[(stages["Data asli"] > 0) | (stages["Setelah pembersihan"] > 0)]
    if len(stages):
        st.dataframe(stages)
        fig = px.bar(stages.reset_index(names="Kolom").melt(id_vars="Kolom", var_name="Tahap", value_name="Missing"),
                     x="Kolom", y="Missing", color="Tahap", barmode="group",
                     title="Missing Values per Kolom: Data Asli vs Pembersihan vs Imputasi")
        fig.update_layout(xaxis_title="Kolom", yaxis_title="Jumlah missing", legend_title="Tahap")
        show_fig(fig, "prep_missing_before_after")
        info_box(f"Total missing: {int(stages['Data asli'].sum())} sel pada data asli, {int(stages['Setelah pembersihan'].sum())} setelah pembersihan "
                 f"(nilai tidak valid dijadikan missing; baris dengan target tidak valid dihapus), dan {int(stages['Setelah imputasi'].sum())} setelah imputasi. "
                 "Kolom Exam_Score tidak diimputasi: baris dengan target kosong atau tidak valid dihapus.")
    # (a2) outlier sebelum vs sesudah pembersihan
    o_raw = outlier_table(raw, num_cols + [TARGET]).set_index("Variabel")["Jumlah outlier"]
    o_clean = outlier_table(clean, num_cols + [TARGET]).set_index("Variabel")["Jumlah outlier"]
    od = pd.DataFrame({"Data asli": o_raw, "Setelah pembersihan": o_clean})
    fig = px.bar(od.reset_index(names="Variabel").melt(id_vars="Variabel", var_name="Tahap", value_name="Jumlah outlier"),
                 x="Variabel", y="Jumlah outlier", color="Tahap", barmode="group", title="Jumlah Outlier IQR: Sebelum vs Sesudah Pembersihan")
    fig.update_layout(xaxis_title="Variabel", yaxis_title="Jumlah outlier", legend_title="Tahap")
    show_fig(fig, "prep_outlier_before_after")
    info_box("Outlier IQR sengaja dipertahankan; perubahan jumlahnya hanya berasal dari nilai tidak valid yang dibersihkan "
             "(mis. Exam_Score atau Attendance di luar rentang logis) dan dari perubahan kuartil akibat baris yang dihapus.")
    # (b) distribusi
    col = st.selectbox("Bandingkan distribusi variabel numerik", num_cols, key="prep_cmp_col")
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{col} - sebelum (training)", f"{col} - sesudah scaling (training)"))
    fig.add_trace(go.Histogram(x=X_tr[col], name="Sebelum", nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=Xt_tr_df[col], name="Sesudah", nbinsx=30), row=1, col=2)
    fig.update_xaxes(title_text="Nilai asli", row=1, col=1)
    fig.update_xaxes(title_text=f"Nilai ({snap['scaler']})", row=1, col=2)
    fig.update_yaxes(title_text="Jumlah siswa", row=1, col=1)
    fig.update_layout(title=f"Distribusi {col}: Sebelum vs Sesudah Scaling", legend_title="Tahap")
    show_fig(fig, f"prep_scaling_{col}")
    info_box("Bentuk distribusi tidak berubah; hanya skala sumbu yang diseragamkan sehingga model berbasis jarak (KNN, SVM) tidak didominasi variabel berskala besar.")
    # (c) dimensi
    dims = pd.DataFrame({"Tahap": ["Sebelum encoding", "Sesudah One-Hot Encoding"], "Jumlah fitur": [X_tr.shape[1], Xt_tr.shape[1]]})
    fig = px.bar(dims, x="Tahap", y="Jumlah fitur", text="Jumlah fitur", color="Tahap", title="Jumlah Fitur: Sebelum vs Sesudah Encoding")
    fig.update_layout(xaxis_title="Tahap", yaxis_title="Jumlah fitur", showlegend=False)
    show_fig(fig, "prep_feature_count")
    info_box("One-Hot Encoding menambah jumlah kolom karena tiap kategori menjadi kolom biner sendiri.")
