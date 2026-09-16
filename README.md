# Telco Customer Churn Prediction — AI Product & MLOps Case Study for Dicoding

> **Membangun sistem AI untuk mengidentifikasi risiko customer churn dari data pelanggan, lalu membawa model tersebut dari tahap eksperimen hingga deployment dan monitoring.**

## Ringkasan Project

Project ini merupakan implementasi **AI Product + MLOps** untuk membantu perusahaan telekomunikasi mengidentifikasi pelanggan yang memiliki risiko tinggi untuk berhenti berlangganan (*customer churn*).

Alih-alih hanya membuat model machine learning dan mengukur akurasinya di notebook, project ini dirancang sebagai **end-to-end AI system**: mulai dari memahami masalah bisnis, mengolah dan memvalidasi data, melatih serta mengevaluasi model, melakukan hyperparameter tuning, menyediakan model melalui REST API, melakukan deployment ke cloud, hingga memantau service setelah berjalan.

Dengan pendekatan ini, output model bukan sekadar angka prediksi, tetapi sebuah **risk signal** yang berpotensi digunakan oleh tim product, marketing, atau customer success sebagai salah satu input untuk menentukan strategi retention.

---

## 1. Masalah Produk

Customer churn merupakan salah satu tantangan penting dalam industri telekomunikasi.

Ketika pelanggan berhenti berlangganan, perusahaan tidak hanya kehilangan pendapatan dari pelanggan tersebut, tetapi juga kehilangan kesempatan untuk mempertahankan hubungan jangka panjang dengan mereka.

Masalahnya, tim bisnis biasanya mengetahui bahwa seorang pelanggan akan churn setelah pelanggan tersebut benar-benar berhenti menggunakan layanan.

Machine learning dapat digunakan untuk mengubah pendekatan ini menjadi lebih proaktif:

```text
Data pelanggan
      ↓
Analisis pola perilaku
      ↓
Prediksi risiko churn
      ↓
Identifikasi pelanggan berisiko
      ↓
Intervensi retention
```

Dengan demikian, pertanyaan produk yang ingin dijawab adalah:

> **“Dari data pelanggan yang tersedia, siapa yang menunjukkan kemungkinan lebih tinggi untuk berhenti berlangganan?”**

Prediksi ini bukan keputusan otomatis untuk memberikan perlakuan tertentu kepada pelanggan. Model berfungsi sebagai **decision-support signal** yang dapat digunakan bersama konteks bisnis dan informasi pelanggan lainnya.

---

## 2. Product Opportunity

Dari sisi product management, peluangnya bukan sekadar membuat model dengan akurasi tinggi.

Model harus mampu menjadi bagian dari sebuah sistem yang dapat digunakan secara nyata.

Karena itu, project ini memiliki beberapa kebutuhan:

* menghasilkan probabilitas churn untuk setiap pelanggan;
* memiliki metrik keberhasilan yang dapat diukur;
* menggunakan preprocessing yang konsisten antara training dan inference;
* memiliki proses validasi data dan model;
* dapat diakses melalui API;
* dapat di-deploy ke environment cloud;
* dapat dipantau setelah deployment;
* dan memiliki struktur pipeline yang dapat direproduksi.

Dengan kata lain, fokus project bergeser dari:

> **“Can we build a model?”**

menjadi:

> **“Can we turn this model into a usable and observable AI capability?”**

---

## 3. Dataset

Project menggunakan dataset **IBM Telco Customer Churn** yang berisi **7.032 records pelanggan**.

Dataset mencakup **12 fitur + 1 label**, termasuk:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`
* `SeniorCitizen`
* `gender`
* `Partner`
* `Dependents`
* `PhoneService`
* `InternetService`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `Churn` sebagai label

Fitur-fitur tersebut merepresentasikan kombinasi informasi pelanggan, layanan yang digunakan, karakteristik kontrak, serta biaya dan lama berlangganan.

---

## 4. Solusi AI

Solusi yang dibangun adalah model **Deep Neural Network (DNN)** untuk memprediksi probabilitas customer churn.

Secara konseptual:

```text
Customer Profile
       +
Service Information
       +
Billing / Tenure
       ↓
Feature Transformation
       ↓
DNN Model
       ↓
Churn Probability
```

Output model berupa probabilitas antara `0` hingga `1`.

Sebagai contoh:

```text
Pelanggan A
Churn probability: 87.93%
→ risiko churn tinggi

Pelanggan B
Churn probability: 1.20%
→ risiko churn rendah
```

Nilai tersebut dapat menjadi input bagi proses pengambilan keputusan berikutnya, misalnya untuk menentukan pelanggan mana yang perlu dianalisis lebih lanjut oleh tim retention.

---

## 5. Data Processing & Feature Engineering

Data diproses menggunakan **TensorFlow Transform (TFT)** agar transformasi yang digunakan saat training dapat direplikasi secara konsisten pada data inference.

### Numerical Features

Fitur:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

dinormalisasi menggunakan **Z-score normalization** melalui:

```python
tft.scale_to_z_score
```

### Binary Feature

`SeniorCitizen` dipertahankan sebagai fitur binary karena nilainya sudah direpresentasikan sebagai `0/1`.

### Categorical Features

Fitur seperti:

* `gender`
* `Partner`
* `Dependents`
* `PhoneService`
* `InternetService`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`

diproses menggunakan vocabulary encoding melalui:

```python
tft.compute_and_apply_vocabulary
```

### Label

Label `Churn` dikonversi:

```text
Yes → 1
No  → 0
```

Data kemudian dipisahkan menjadi **80% training dan 20% evaluation** menggunakan ExampleGen dalam pipeline TFX.

---

## 6. Model & Hyperparameter Tuning

Model menggunakan arsitektur Deep Neural Network dengan struktur:

```text
Input
  ↓
Dense(128, ReLU)
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU)
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(1, Sigmoid)
```

Model menggunakan:

* **Optimizer:** Adam
* **Loss:** Binary Crossentropy
* **Output:** Sigmoid
* **Hyperparameter tuning:** Keras Tuner RandomSearch

Tuning dijalankan melalui komponen **TFX Tuner**, sehingga proses pemilihan konfigurasi model menjadi bagian dari pipeline machine learning.

---

## 7. Success Metrics

Dua metrik utama digunakan untuk mengevaluasi model:

### Binary Accuracy

Target:

```text
≥ 0.78
```

Metrik ini digunakan untuk melihat proporsi prediksi klasifikasi yang benar secara keseluruhan.

### AUC

Target:

```text
≥ 0.75
```

AUC digunakan untuk mengevaluasi kemampuan model dalam membedakan pelanggan yang churn dan tidak churn pada berbagai threshold.

Evaluasi dilakukan menggunakan **TensorFlow Model Analysis (TFMA)**.

---

## 8. Hasil Model

Hasil evaluasi:

| Metrik          | Target |     Hasil |
| --------------- | -----: | --------: |
| Binary Accuracy | ≥ 0.78 | **0.808** |
| AUC             | ≥ 0.75 | **0.870** |

Kedua metrik tersebut melewati threshold yang ditentukan untuk project.

Pada contoh inference yang digunakan dalam project, model juga menghasilkan perbedaan probabilitas yang cukup jelas antara sample pelanggan berisiko tinggi dan rendah:

```text
High-risk sample
Churn probability: 87.93%

Low-risk sample
Churn probability: 1.20%
```

Angka probabilitas ini bukan berarti pelanggan tersebut pasti atau tidak akan churn. Probabilitas digunakan sebagai **risk signal** untuk membantu proses analisis dan pengambilan keputusan.

---

# 9. MLOps Pipeline

Setelah model berhasil dibuat, tantangan berikutnya adalah bagaimana membuatnya dapat dijalankan secara konsisten.

Project menggunakan **TensorFlow Extended (TFX)** dengan **Apache Beam / BeamDagRunner** untuk membangun pipeline:

```text
CSV Data
   ↓
CsvExampleGen
   ↓
StatisticsGen
   ↓
SchemaGen
   ↓
ExampleValidator
   ↓
Transform
   ↓
Tuner
   ↓
Trainer
   ↓
Evaluator / TFMA
   ↓
Pusher
   ↓
TensorFlow Serving
```

Pendekatan ini membuat proses machine learning tidak berhenti di notebook, tetapi menjadi rangkaian workflow yang lebih terstruktur dan dapat direproduksi.

---

# 10. Deployment

Model kemudian di-serving menggunakan:

**TensorFlow Serving 2.11.0**

Model dikemas menggunakan **Docker** dan di-deploy ke **Railway**.

Arsitekturnya:

```text
Trained Model
     ↓
TensorFlow Serving
     ↓
Docker Container
     ↓
Railway
     ↓
REST API
```

Model dapat diakses melalui endpoint REST API:

**Churn Model API**

https://kendrickfff-submission2-production.up.railway.app/v1/models/churn-model

Dengan pendekatan ini, model dapat dipanggil oleh aplikasi atau service lain tanpa perlu menjalankan notebook training.

---

# 11. Monitoring & Observability

Deployment bukan akhir dari lifecycle model.

Setelah model berjalan sebagai service, kita tetap perlu mengetahui apakah service tersebut berjalan dengan baik.

Untuk itu project menggunakan:

* **Prometheus** untuk mengumpulkan metrics;
* **Grafana** untuk visualisasi monitoring;
* **Docker Compose** untuk menjalankan monitoring stack secara lokal.

Metrics yang dipantau mencakup:

* request count;
* request latency;
* model status;
* service availability.

Arsitekturnya:

```text
TensorFlow Serving
       ↓
   Metrics
       ↓
  Prometheus
       ↓
    Grafana
       ↓
Monitoring Dashboard
```

Dashboard digunakan untuk memberikan visibilitas terhadap kondisi service setelah deployment.

---

# 12. Dari Perspektif AI Product Management

Hal yang ingin ditunjukkan dari project ini bukan hanya kemampuan membangun model machine learning.

Project ini memperlihatkan bagaimana sebuah **AI capability** dapat dikembangkan dari masalah bisnis sampai menjadi service yang dapat digunakan.

```text
Business Problem
      ↓
Customer Churn
      ↓
Product Opportunity
      ↓
AI Use Case
      ↓
Success Metrics
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Production API
      ↓
Deployment
      ↓
Monitoring
      ↓
Business Decision Support
```

Dari perspektif AI Product Management, beberapa keputusan penting dalam project meliputi:

**Problem framing**
Mengubah masalah customer churn menjadi use case prediksi risiko.

**Success metrics**
Menentukan threshold evaluasi sebelum menilai hasil model.

**AI-product integration**
Memikirkan bagaimana output model dapat digunakan sebagai input bagi proses bisnis, bukan hanya menghasilkan angka di notebook.

**Technical feasibility**
Menggunakan TFX, TensorFlow Serving, Docker, dan cloud deployment untuk membawa model menuju production-like environment.

**Operational readiness**
Menambahkan monitoring agar service dapat diamati setelah deployment.

**Responsible interpretation**
Memperlakukan churn probability sebagai risk signal, bukan sebagai keputusan otomatis terhadap pelanggan.

---

# 13. Tech Stack

| Area                     | Technology                |
| ------------------------ | ------------------------- |
| Programming              | Python                    |
| ML Framework             | TensorFlow / Keras        |
| MLOps                    | TensorFlow Extended (TFX) |
| Orchestration            | Apache Beam               |
| Feature Engineering      | TensorFlow Transform      |
| Hyperparameter Tuning    | Keras Tuner               |
| Model Evaluation         | TensorFlow Model Analysis |
| Model Serving            | TensorFlow Serving 2.11.0 |
| Containerization         | Docker                    |
| Deployment               | Railway                   |
| API                      | REST API                  |
| Monitoring               | Prometheus                |
| Visualization            | Grafana                   |
| Experiment / Development | Jupyter Notebook          |

---

# 14. Project Takeaway

Project ini mendemonstrasikan pendekatan **end-to-end AI product development**:

> **Memulai dari masalah bisnis, menerjemahkannya menjadi AI use case, membangun dan mengevaluasi model, membawa model menjadi production service, kemudian memonitor sistem setelah deployment.**

Dengan demikian, machine learning tidak diposisikan sebagai fitur yang berdiri sendiri, tetapi sebagai bagian dari **produk dan sistem yang lebih besar**.

**Business problem → AI capability → production system → measurable outcome.**
