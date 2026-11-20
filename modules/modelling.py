"""Menu D - Modelling: pelatihan dan perbandingan lima algoritma klasifikasi."""
import time

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modules.classification import get_thresholds
from modules.data_loader import CLASS_ORDER, MODEL_DIR, TARGET, ensure_output_dirs, info_box, save_table, show_fig, split_columns
from modules.preprocessing import build_preprocessor, split_data

ALGORITHMS = ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine"]
METRIC_OPTIONS = {
    "F1-Score (weighted)": "F1 (weighted)",
    "Accuracy": "Accuracy",
    "F1-Score (macro)": "F1 (macro)",
    "Precision (weighted)": "Precision (weighted)",
    "Recall (weighted)": "Recall (weighted)",
}


def _param_ui(name: str, seed: int, balanced: bool):
    """Widget parameter per algoritma; kembalikan (estimator, dict parameter)."""
    cw = "balanced" if balanced else None
    k = name.replace(" ", "_")
    if name == "Logistic Regression":
        C = st.number_input("C (kebalikan regularisasi)", 0.001, 100.0, 1.0, 0.1, key=f"{k}_C")
        it = st.number_input("max_iter", 100, 5000, 1000, 100, key=f"{k}_it")
        return LogisticRegression(C=C, max_iter=int(it), class_weight=cw, random_state=seed), {"C": C, "max_iter": int(it)}
    if name == "Decision Tree":
        crit = st.selectbox("criterion", ["gini", "entropy"], key=f"{k}_crit")
        depth = st.slider("max_depth (0 = tanpa batas)", 0, 30, 5, key=f"{k}_depth")
        mss = st.slider("min_samples_split", 2, 50, 2, key=f"{k}_mss")
        return (DecisionTreeClassifier(criterion=crit, max_depth=depth or None, min_samples_split=mss, class_weight=cw, random_state=seed),
                {"criterion": crit, "max_depth": depth or None, "min_samples_split": mss})
    if name == "Random Forest":
        n = st.slider("n_estimators", 10, 500, 100, 10, key=f"{k}_n")
        depth = st.slider("max_depth (0 = tanpa batas)", 0, 30, 0, key=f"{k}_depth")
        mss = st.slider("min_samples_split", 2, 50, 2, key=f"{k}_mss")
        return (RandomForestClassifier(n_estimators=n, max_depth=depth or None, min_samples_split=mss, class_weight=cw, random_state=seed, n_jobs=-1),
                {"n_estimators": n, "max_depth": depth or None, "min_samples_split": mss})
    if name == "K-Nearest Neighbors":
        nn = st.slider("n_neighbors", 1, 50, 5, key=f"{k}_nn")
        w = st.selectbox("weights", ["uniform", "distance"], key=f"{k}_w")
        return KNeighborsClassifier(n_neighbors=nn, weights=w, n_jobs=-1), {"n_neighbors": nn, "weights": w}
    if name == "Support Vector Machine":
        C = st.number_input("C", 0.01, 100.0, 1.0, 0.1, key=f"{k}_C")
        ker = st.selectbox("kernel", ["rbf", "linear", "poly"], key=f"{k}_ker")
        gam = st.selectbox("gamma", ["scale", "auto"], key=f"{k}_gam", disabled=(ker == "linear"))
        return SVC(C=C, kernel=ker, gamma=gam, class_weight=cw, random_state=seed), {"C": C, "kernel": ker, "gamma": gam}
    raise ValueError(name)


def _metrics(y_true, y_pred) -> dict:
    pw = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
    pm = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": pw[0], "Recall (weighted)": pw[1], "F1 (weighted)": pw[2],
        "Precision (macro)": pm[0], "Recall (macro)": pm[1], "F1 (macro)": pm[2],
    }


def results_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results["models"].items():
        rows.append({"Algoritma": name, **r["metrics"], "Train Accuracy": r["train_accuracy"], "Waktu latih (detik)": r["fit_seconds"]})
    return pd.DataFrame(rows).set_index("Algoritma")


def stale_reasons(results: dict) -> list[str]:
    """Bandingkan pengaturan saat training dengan pengaturan aktif; kembalikan daftar perbedaan."""
    meta, reasons = results["meta"], []
    if "clean_df" not in st.session_state:
        return ["data hasil preprocessing sudah direset"]
    cur = {
        "test size": st.session_state.get("cfg_test_size", 0.2),
        "random state": st.session_state.get("cfg_random_state", 42),
        "threshold kelas": tuple(st.session_state.get("cfg_thresholds", meta["thresholds"])),
        "pengaturan preprocessing": st.session_state.get("prep_snapshot"),
    }
    old = {"test size": meta["test_size"], "random state": meta["random_state"],
           "threshold kelas": tuple(meta["thresholds"]), "pengaturan preprocessing": meta["prep"]}
    for k in cur:
        if cur[k] != old[k]:
            reasons.append(k)
    if meta.get("n_rows") is not None and meta["n_rows"] != len(st.session_state["clean_df"]):
        reasons.append("jumlah data bersih")
    return reasons


def stale_warning(results: dict) -> None:
    r = stale_reasons(results)
    if r:
        st.warning("⚠️ Hasil di bawah berasal dari pengaturan lama; berbeda pada: " + ", ".join(r)
                   + ". Klik **Jalankan Training** lagi agar hasil sesuai pengaturan sekarang.")


def best_model_name(results: dict, metric: str) -> str:
    return results_table(results)[metric].idxmax()


def render(df: pd.DataFrame) -> None:
    st.header("🤖 Modelling")
    if "clean_df" not in st.session_state:
        st.warning("Jalankan **Data Preprocessing** terlebih dahulu.")
        return

    clean = st.session_state["clean_df"]
    num_cols, cat_cols = split_columns(clean)
    prep_cfg = st.session_state["prep_snapshot"]
    labels = st.session_state.get("class_labels")
    if labels is None:
        from modules.classification import assign_classes
        labels = assign_classes(clean[TARGET], *get_thresholds())
    low_max, high_min = get_thresholds()

    st.info(f"Target: kelas performa (Low < {low_max}, High >= {high_min}). Fitur: seluruh variabel prediktor — "
            f"`{TARGET}` **tidak** dipakai sebagai fitur untuk mencegah data leakage.")

    with st.container(border=True):
        st.subheader("Pengaturan")
        a, b, c = st.columns(3)
        test_size = a.slider("Test size", 0.1, 0.5, float(st.session_state.get("cfg_test_size", 0.2)), 0.05)
        seed = int(b.number_input("Random state", 0, 99999, int(st.session_state.get("cfg_random_state", 42)), 1))
        balanced = c.checkbox("class_weight = 'balanced' (LR, DT, RF, SVM)", value=False)
        st.session_state["cfg_test_size"], st.session_state["cfg_random_state"] = test_size, seed
        selected = st.multiselect("Algoritma yang dilatih & dibandingkan", ALGORITHMS, default=ALGORITHMS)
        metric_label = st.selectbox("Metrik penentu model terbaik", list(METRIC_OPTIONS))
        metric = METRIC_OPTIONS[metric_label]

        st.markdown("**Parameter model**")
        estimators, params = {}, {}
        for name in selected:
            with st.expander(name):
                estimators[name], params[name] = _param_ui(name, seed, balanced)

    if st.button("▶️ Jalankan Training", type="primary", disabled=not selected):
        _train(clean, labels, num_cols, cat_cols, prep_cfg, estimators, params, test_size, seed, metric, balanced)

    results = st.session_state.get("results")
    if results is None:
        st.info("Belum ada model yang dilatih. Atur parameter lalu klik **Jalankan Training**.")
        return
    _show_results(results, metric)


def _train(clean, labels, num_cols, cat_cols, prep_cfg, estimators, params, test_size, seed, metric, balanced):
    try:
        X_tr, X_te, y_tr, y_te = split_data(clean, labels, test_size, seed)
    except ValueError as e:
        st.error(f"Train-test split gagal: {e}")
        return
    pre = build_preprocessor(num_cols, cat_cols, prep_cfg["num_strategy"], prep_cfg["scaler"])
    models = {}
    bar = st.progress(0.0, text="Memulai training...")
    try:
        for i, (name, est) in enumerate(estimators.items()):
            bar.progress(i / len(estimators), text=f"Melatih {name}...")
            pipe = Pipeline([("prep", clone(pre)), ("model", est)])
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            y_pred = pipe.predict(X_te)
            models[name] = {
                "pipeline": pipe,
                "params": params[name],
                "y_pred": pd.Series(y_pred, index=y_te.index),
                "metrics": _metrics(y_te, y_pred),
                "train_accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
                "cm": confusion_matrix(y_te, y_pred, labels=CLASS_ORDER),
                "report": classification_report(y_te, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0),
                "fit_seconds": fit_s,
            }
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Training gagal: {e}")
        return
    bar.progress(1.0, text="Training selesai")

    results = {
        "models": models, "y_test": y_te, "y_train": y_tr,
        "meta": {"test_size": test_size, "random_state": seed, "thresholds": get_thresholds(),
                 "n_train": len(X_tr), "n_test": len(X_te), "n_rows": len(clean), "balanced": balanced, "prep": dict(prep_cfg)},
    }
    st.session_state["results"] = results

    try:  # simpan artefak (opsional, kegagalan tidak boleh membatalkan hasil)
        ensure_output_dirs()
        tbl = results_table(results)
        save_table(tbl.round(4), "model_comparison")
        best = best_model_name(results, metric)
        joblib.dump({"pipeline": models[best]["pipeline"], "name": best, "classes": CLASS_ORDER, "meta": results["meta"]},
                    MODEL_DIR / "best_model.joblib", compress=3)
        st.success(f"Training selesai. Model terbaik berdasarkan {metric}: **{best}** (disimpan ke outputs/models/best_model.joblib).")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Model berhasil dilatih, tetapi penyimpanan file gagal: {e}")


def _show_results(results: dict, metric: str) -> None:
    meta = results["meta"]
    stale_warning(results)
    st.caption(f"Hasil training terakhir: test size {meta['test_size']:.0%}, random state {meta['random_state']}, "
               f"{meta['n_train']:,} data training / {meta['n_test']:,} data testing, "
               f"threshold {meta['thresholds'][0]} / {meta['thresholds'][1]}.")
    tbl = results_table(results)
    best = best_model_name(results, metric)
    st.subheader("Perbandingan algoritma")
    st.success(f"Model terbaik berdasarkan **{metric}**: **{best}** ({tbl.loc[best, metric]:.4f})")
    show_cols = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "F1 (macro)", "Train Accuracy", "Waktu latih (detik)"]
    st.dataframe(tbl[show_cols].sort_values(metric, ascending=False).style.format("{:.4f}").highlight_max(subset=show_cols[:5], color="#c6f6d5"))

    m = tbl[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]].reset_index().melt(
        id_vars="Algoritma", var_name="Metrik", value_name="Skor")
    fig = px.bar(m, x="Algoritma", y="Skor", color="Metrik", barmode="group", title="Perbandingan Performa Antaralgoritma (Data Testing)")
    fig.update_layout(xaxis_title="Algoritma", yaxis_title="Skor", yaxis_range=[0, 1], legend_title="Metrik")
    show_fig(fig, "model_comparison_metrics")
    spread = tbl[metric].max() - tbl[metric].min()
    info_box(f"{best} memperoleh {metric} tertinggi. Selisih antara algoritma terbaik dan terendah adalah {spread:.4f}"
             + (" — sangat kecil, jadi perbedaan antaralgoritma mungkin tidak bermakna." if spread < 0.02 else "."))

    a, b = st.columns(2)
    with a:
        g = tbl[["Train Accuracy", "Accuracy"]].rename(columns={"Accuracy": "Test Accuracy"}).reset_index().melt(
            id_vars="Algoritma", var_name="Data", value_name="Accuracy")
        fig = px.bar(g, x="Algoritma", y="Accuracy", color="Data", barmode="group", title="Akurasi Training vs Testing")
        fig.update_layout(xaxis_title="Algoritma", yaxis_title="Accuracy", yaxis_range=[0, 1.05], legend_title="Data")
        show_fig(fig, "model_train_vs_test_accuracy")
    with b:
        fig = px.bar(tbl.reset_index(), x="Algoritma", y="Waktu latih (detik)", text=tbl["Waktu latih (detik)"].round(2).values,
                     title="Waktu Training per Algoritma")
        fig.update_layout(xaxis_title="Algoritma", yaxis_title="Detik")
        show_fig(fig, "model_fit_time")
    gap = (tbl["Train Accuracy"] - tbl["Accuracy"]).sort_values(ascending=False)
    info_box(f"Selisih akurasi training dan testing terbesar ada pada {gap.index[0]} ({gap.iloc[0]:.3f}). "
             "Selisih besar menandakan model menghafal data training (overfitting). Detail metrik per kelas ada di menu Evaluation.")
