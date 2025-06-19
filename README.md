# Proyek Sistem Rekomendasi (Content-Based & Collaborative Filtering)

## 📜 Deskripsi Proyek

Proyek ini bertujuan untuk membangun dan membandingkan dua pendekatan utama dalam sistem rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering**. Sistem ini dirancang untuk memberikan rekomendasi item (misalnya film, buku, atau produk) yang relevan kepada pengguna. Proyek ini merupakan bagian dari submission kelas **Machine Learning Terapan** di Dicoding.

## 📌 Latar Belakang

Dalam platform digital dengan jutaan item, pengguna sering kesulitan menemukan konten yang mereka sukai. Sistem rekomendasi membantu mempersonalisasi pengalaman pengguna, meningkatkan engagement, dan mendorong penemuan konten baru.

## 📊 Dataset

Dataset yang digunakan dalam proyek ini berisi informasi mengenai item (misal: judul, genre, deskripsi) dan data interaksi pengguna (misal: rating yang diberikan pengguna terhadap item).
_(Spesifikasikan nama dataset yang Anda gunakan di sini, contoh: MovieLens Dataset)._

## 🛠️ Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama.
- **Pandas & NumPy**: Untuk analisis dan manipulasi data.
- **Scikit-learn**: Terutama untuk `TfidfVectorizer` pada Content-Based Filtering.
- **Matplotlib/Seaborn**: Untuk visualisasi data.
- **Jupyter Notebook**: Sebagai lingkungan pengembangan.

## ⚙️ Tahapan Proyek

1.  **Data Understanding & EDA**: Memahami karakteristik data item dan interaksi pengguna.
2.  **Data Preparation**: Membersihkan data dan menyiapkannya untuk kedua pendekatan model.
3.  **Model Development - Content-Based Filtering**:
    - Membangun sistem yang merekomendasikan item berdasarkan kemiripan atributnya (misalnya, merekomendasikan film dengan genre yang sama).
    - Menggunakan TF-IDF Vectorizer untuk mengubah fitur teks menjadi vektor dan Cosine Similarity untuk mengukur kemiripan.
4.  **Model Development - Collaborative Filtering**:
    - Membangun sistem yang merekomendasikan item berdasarkan preferensi pengguna yang memiliki selera serupa.
    - Mengimplementasikan algoritma berbasis kemiripan pengguna atau item.
5.  **Evaluation**: Mengevaluasi kualitas rekomendasi menggunakan metrik presisi.

## ✨ Hasil Utama

- Berhasil mengimplementasikan dua jenis sistem rekomendasi yang fundamental.
- Mampu menyajikan top-N item yang paling relevan untuk seorang pengguna berdasarkan histori interaksinya atau kemiripan item.
- Memahami kelebihan dan kekurangan dari masing-masing pendekatan (misalnya, _cold start problem_ pada Collaborative Filtering).

## 🚀 Cara Menjalankan Proyek

1.  **Clone Repositori**:
    ```bash
    git clone [https://github.com/leesnar/proyek-sistem-rekomendasi.git](https://github.com/leesnar/proyek-sistem-rekomendasi.git)
    ```
2.  **Jalankan Notebook**: Buka dan jalankan file `.ipynb` yang tersedia di dalam repositori ini menggunakan Jupyter Notebook atau Google Colab.
