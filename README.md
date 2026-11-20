# Student Performance Analysis (Streamlit)

Dashboard analisis dan klasifikasi performa siswa dari `StudentPerformanceFactors.csv`.

## Instalasi
1. Pastikan `StudentPerformanceFactors.csv` ada di folder proyek ini (sejajar dengan `app.py`); sudah disertakan di zip.
2. (Disarankan) buat virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan
```bash
streamlit run app.py
```
Buka alamat yang tampil di terminal (biasanya http://localhost:8501).

## Alur & fungsi menu
Ikuti urutan menu di sidebar:

| Menu | Fungsi |
|---|---|
| Overview | Ringkasan dataset asli: ukuran, tipe data, missing, duplikat, statistik, histogram/boxplot/bar chart, heatmap korelasi. |
| Data Preprocessing | Diagnosis missing/duplikat/outlier/nilai tidak valid; pembersihan pada salinan data; imputasi, One-Hot Encoding, scaling lewat `ColumnTransformer`; stratified train-test split; perbandingan sebelum vs sesudah. |
| Classification | Membentuk kelas Low/Medium/High dari `Exam_Score` dengan threshold yang dapat diatur; distribusi kelas dan perbandingan training vs testing. |
| Modelling | Memilih algoritma (Logistic Regression, Decision Tree, Random Forest, KNN, SVM), test size, random state, parameter; training dan perbandingan model. |
| Evaluation | Accuracy, Precision, Recall, F1, confusion matrix, classification report, perbandingan antaralgoritma, dan interpretasi otomatis. |

## Keputusan desain penting
- `Exam_Score` tidak pernah menjadi fitur model (hanya dasar pembentukan kelas).
- Imputasi, scaling, dan encoding di-fit hanya pada data training (tanpa data leakage).
- Nilai fitur tidak valid (mis. Attendance di luar 0-100) menjadi missing lalu diimputasi; baris dengan `Exam_Score` di luar 0-100 dihapus. Outlier IQR hanya ditandai, tidak dibuang.
- Threshold default = persentil 33,3 dan 66,7 dari data bersih; dapat diubah di menu Classification.
- Hasil proses disimpan di `st.session_state`; tabel, figur HTML, dan `best_model.joblib` ditulis ke `outputs/`.
