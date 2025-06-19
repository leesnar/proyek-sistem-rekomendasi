# Laporan Proyek Machine Learning - Lesnar Tambun MC006D5Y2425

## Project Overview
Platform streaming seperti Netflix menghadapi tantangan **information overload**, di mana pengguna kesulitan menemukan film relevan karena banyaknya pilihan konten. Hal ini dapat menurunkan kepuasan, retensi, dan pendapatan platform. Sistem rekomendasi hybrid yang menggabungkan *content-based filtering* dan *collaborative filtering* menjadi solusi untuk memberikan rekomendasi personal dan relevan.

**Proyek ini bertujuan membangun sistem rekomendasi hybrid menggunakan dataset MovieLens untuk membantu pengguna menemukan film sesuai preferensi, mengatasi *cold start*, dan meningkatkan pengalaman pengguna.**

## Business Understanding
### Problem Statements
1. Pengguna kesulitan menemukan film relevan, menurunkan kepuasan dan retensi.  
2. Sistem rekomendasi tunggal sering gagal menangani *cold start* atau memberikan rekomendasi sempit berdasarkan metadata.

### Goals
1. Membangun sistem rekomendasi hybrid yang menggabungkan *content-based filtering* (berbasis genre dan tag) dan *collaborative filtering* (berbasis rating) untuk rekomendasi akurat.  
2. Mengatasi *cold start* dan meningkatkan variasi rekomendasi.

### Solution Approach
1. **Content-Based Filtering**: Menggunakan TF-IDF Vectorizer untuk genre dan tag, dengan cosine similarity untuk kesamaan film.  
2. **Collaborative Filtering**: Menggunakan Singular Value Decomposition (SVD) dari library Surprise untuk prediksi rating.  
3. **Hybrid System**: Menggabungkan skor kedua metode (bobot 0.7:0.3) untuk rekomendasi optimal.

## Data Understanding
**Dataset**: MovieLens ml-latest-small dari GroupLens (https://grouplens.org/datasets/movielens/).  
- **movies.csv**: 9.742 film.  
- **ratings.csv**: 100.836 rating dari 610 pengguna.  
- **tags.csv**: 3.683 tag pengguna.  

### Jumlah dan Kondisi Data
- **movies.csv**: Berisi movieId, title, genres; tanpa *missing value* atau duplikat.  
- **ratings.csv**: Berisi userId, movieId, rating (0.5-5.0), timestamp; tanpa *missing value* atau duplikat.  
- **tags.csv**: Berisi userId, movieId, tag, timestamp; tanpa *missing value* atau duplikat.

### Variabel/Fitur
- **movies.csv**:  
  - movieId: ID unik film (integer).  
  - title: Judul dan tahun rilis (string, contoh: "Toy Story (1995)").  
  - genres: Genre, dipisah | (string, contoh: "Adventure|Animation|Children|Comedy|Fantasy").  
- **ratings.csv**:  
  - userId: ID pengguna (integer).  
  - movieId: ID film (integer).  
  - rating: Nilai rating (float, 0.5-5.0).  
  - timestamp: Waktu rating (integer, UNIX).  
- **tags.csv**:  
  - userId: ID pengguna (integer).  
  - movieId: ID film (integer).  
  - tag: Kata kunci (string, contoh: "funny").  
  - timestamp: Waktu tag (integer).  

### Exploratory Data Analysis (EDA)
**Temuan Utama**:  
- **Distribusi Rating**: Rata-rata ~3.5, condong ke 3-5, menunjukkan preferensi positif.  
- **Jumlah Rating**: Banyak pengguna memberikan <200 rating, banyak film memiliki <50 rating (sparsity tinggi).  
- **Genre Populer**: Drama, Comedy, Action.  
- **Top-10 Most Rated Movies**: Forrest Gump (329 rating), Shawshank Redemption (317), Pulp Fiction (307), dll.  
**Insight**: Sparsity memerlukan filtering data untuk modeling.

![EDA](https://imgur.com/ZmJBIBs.png "EDA")
**Gambar 1**: Exploratory Data Analysis
## Data Preparation
### Teknik Data Preparation
1. **Pemeriksaan Kualitas Data**:  
   - Memeriksa *missing value*: Tidak ada pada `movies.csv`, `ratings.csv`, `tags.csv`.  
   - Memeriksa duplikat: Tidak ada duplikat.  
   - **Alasan**: Memastikan data bersih untuk analisis dan modeling.

2. **Filtering Sparsity**:  
   - Menghapus pengguna dengan <10 rating dan film dengan <20 rating, menghasilkan 67.898 rating.  
   - **Alasan**: Mengurangi sparsity matriks rating, meningkatkan akurasi *collaborative filtering*.

3. **Pembuatan Matriks User-Movie**:  
   - Membuat matriks rating (610 pengguna × 1.297 film) dengan `pivot_table`.  
   - Mengisi *missing value* dengan 0 dan mengubah ke sparse matrix (`csr_matrix`).  
   - **Alasan**: Sparse matrix efisien untuk dataset besar dengan sparsity tinggi.

4. **Feature Engineering untuk Content-Based Filtering**:  
   - Membersihkan kolom genres, mengganti nilai kosong dengan string kosong.  
   - Membuat `genre_string` (contoh: "Adventure Animation Children Comedy Fantasy").  
   - Mengintegrasikan tag dari `tags.csv` ke kolom `content` (genre + tag).  
   - Menerapkan **TF-IDF Vectorization** pada `content` (max_features=1677), menghasilkan matriks fitur (9.742 × 1.677).  
   - **Alasan**: TF-IDF mengubah teks menjadi vektor numerik untuk cosine similarity, tag meningkatkan relevansi rekomendasi.

### Justifikasi
- Filtering sparsity menghasilkan matriks rating lebih padat untuk SVD.  
- Sparse matrix mengoptimalkan penggunaan memori.  
- TF-IDF dan tag menciptakan fitur kaya untuk *content-based filtering*.  
- Pemeriksaan kualitas data mencegah error modeling.

## Modeling
### 1. Content-Based Filtering
**Algoritma**: Cosine Similarity.  
**Proses**:  
- Menggunakan matriks TF-IDF untuk menghitung cosine similarity antar film (matriks 1.297 × 1.297).  
- Fungsi `get_content_recommendations` menghasilkan Top-N film berdasarkan judul input.  
**Kelebihan**: Efektif untuk *cold start*, rekomendasi jelas berdasarkan genre/tag.  
**Kekurangan**: Terbatas pada fitur genre dan tag.  
**Contoh Output (Top-5 untuk "Toy Story (1995)")**:  
1. Bug's Life, A (1998) - Adventure|Animation|Children|Comedy  
2. Toy Story 2 (1999) - Adventure|Animation|Children|Comedy|Fantasy  
3. Antz (1998) - Adventure|Animation|Children|Comedy|Fantasy  
4. Emperor's New Groove, The (2000) - Adventure|Animation|Children|Comedy|Fantasy  
5. Monsters, Inc. (2001) - Adventure|Animation|Children|Comedy|Fantasy  

**Contoh Output (Top-5 untuk "Pulp Fiction (1994)")**:  
1. Reservoir Dogs (1992) - Crime|Mystery|Thriller  
2. Big Lebowski, The (1998) - Comedy|Crime  
3. Sin City (2005) - Action|Crime|Film-Noir|Mystery|Thriller  
4. Django Unchained (2012) - Action|Drama|Western  
5. Kiss Kiss Bang Bang (2005) - Comedy|Crime|Mystery|Thriller  

**Contoh Output (Top-5 untuk "Titanic (1997)")**:  
1. Bridges of Madison County, The (1995) - Drama|Romance  
2. Walk in the Clouds, A (1995) - Drama|Romance  
3. Piano, The却 - Drama|Romance  
4. Phenomenon (1996) - Drama|Romance  
5. American Beauty (1999) - Drama|Romance  

### 2. Collaborative Filtering
**Algoritma**: Surprise SVD.  
**Proses**:  
- Membuat matriks user-item dari 67.898 rating.  
- SVD dengan tuning `n_factors` (20, 50, 100) untuk prediksi rating.  
- Fungsi `get_collaborative_recommendations` menghasilkan Top-N film belum dirating.  
**Kelebihan**: Menangkap pola preferensi pengguna serupa.  
**Kekurangan**: Kurang efektif untuk *cold start*.  
**Contoh Output (Top-5 untuk User 1)**:  
1. Shawshank Redemption, The (1994) - Crime|Drama  
2. Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) - Comedy|War  
3. Godfather, The (1972) - Crime|Drama  
4. Philadelphia Story, The (1940) - Comedy|Drama|Romance  
5. Rear Window (1954) - Mystery|Thriller  

### 3. Hybrid System
**Proses**:  
- Menggabungkan skor *content-based* dan *collaborative filtering* (bobot 0.7:0.3).  
- Fungsi `get_hybrid_recommendations` mengurutkan berdasarkan skor gabungan.  
**Kelebihan**: Mengatasi *cold start* dan meningkatkan personalisasi.  
**Kekurangan**: Bergantung pada kualitas kedua model.  
**Contoh Output (Top-8 untuk User 1 dan "The Shawshank Redemption (1994)")**:  
1. Shawshank Redemption, The (1994) - Score: 0.300 - Crime|Drama  
2. Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) - Score: 0.290 - Comedy|War  
3. Godfather, The (1972) - Score: 0.280 - Crime|Drama  
4. Philadelphia Story, The (1940) - Score: 0.270 - Comedy|Drama|Romance  
5. Rear Window (1954) - Score: 0.260 - Mystery|Thriller  
6. North by Northwest (1959) - Score: 0.250 - Action|Adventure|Mystery|Romance|Thriller  
7. Casablanca (1942) - Score: 0.240 - Drama|Romance  
8. One Flew Over the Cuckoo's Nest (1975) - Score: 0.230 - Drama  

## Evaluation

![Evaluation](https://imgur.com/XETjzLC.png "Evaluation")
**Gambar 2**: Evaluation

### Metrik Evaluasi

#### 1. Root Mean Square Error (RMSE) untuk Collaborative Filtering

**Formula:**
```
RMSE = √(1/n × Σ(yi - ŷi)²)
```

Atau dalam notasi matematis:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

- Mengukur rata-rata error kuadrat prediksi rating
- Nilai kecil menunjukkan akurasi tinggi

#### 2. Mean Absolute Error (MAE) untuk Collaborative Filtering

**Formula:**
```
MAE = 1/n × Σ|yi - ŷi|
```

Atau dalam notasi matematis:

$$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

- Mengukur rata-rata error absolut
- Robust terhadap outlier

#### 3. Precision@10 untuk Content-Based dan Hybrid System

**Formula:**
```
Precision@k = Number of relevant recommendations / k
```

Atau dalam notasi matematis:

$$Precision@k = \frac{\text{Number of relevant recommendations}}{k}$$

**Kriteria Relevant:**
- Genre serupa untuk *content-based*
- Rating ≥4 untuk *hybrid*

**Fungsi:** Mengukur proporsi rekomendasi relevan

#### 4. Recall@10 untuk Content-Based

**Formula:**
```
Recall@k = Number of relevant recommendations / |relevant set|
```

Atau dalam notasi matematis:

$$Recall@k = \frac{\text{Number of relevant recommendations}}{|\text{relevant set}|}$$

**Fungsi:** Mengukur proporsi genre relevan yang ditemukan

---

### Hasil Evaluasi
1. **Content-Based Filtering**:  
   - **Precision@10**: 1.0 untuk "Toy Story (1995)", 0.0 untuk "The Matrix (1999)".  
   - **Recall@10**: 2.0 untuk "Toy Story (1995)", 0.0 untuk "The Matrix (1999)".  
   - **Analisis**: Precision@10 tinggi untuk "Toy Story" karena genre animasi seragam, tetapi rendah untuk "The Matrix" karena genre sci-fi lebih beragam. Recall@10 untuk "Toy Story" melebihi 1.0 karena implementasi menghitung duplikasi genre, menunjukkan perlunya normalisasi.

2. **Collaborative Filtering**:  
   - **RMSE**: 0.8395 (n_factors=50).  
   - **MAE**: 0.6423.  
   - **Analisis**: SVD menghasilkan prediksi akurat (error <1.0), dengan `n_factors=50` sebagai parameter optimal berdasarkan tuning. Performa jauh lebih baik dibandingkan implementasi awal (RMSE: 3.1669, MAE: 2.9571).

3. **Hybrid System**:  
   - **Precision@10**: 0.0 untuk User 1, 4, 6.  
   - **Analisis**: Precision@10 rendah karena evaluasi menggunakan pengguna aktif tanpa cukup data relevansi rating ≥4. Filtering sparsity dan bobot 0.7:0.3 belum cukup mengatasi *cold start* pada kasus ini.

## Visualisasi dan Analisis Model
### Tuning Plot
#### Hasil Tuning Hyperparameter
- **RMSE/MAE vs. n_factors** (20, 50, 100) menunjukkan n_factors=50 optimal
- **RMSE**: 0.8395
- **MAE**: 0.6423

**Insight:** Tuning hyperparameter membantu mengidentifikasi konfigurasi terbaik untuk model SVD, dengan error terendah pada n_factors=50.

### Heatmap Cosine Similarity

#### Hasil Analisis
Menunjukkan kesamaan tinggi antar film animasi (contoh: "Toy Story").

**Insight:** Film dengan genre serupa (seperti animasi) memiliki skor cosine similarity tinggi, menunjukkan efektivitas TF-IDF dalam menangkap kesamaan fitur content-based.

### Analisis Subplot Visualisasi
#### 1. Predicted vs Actual Ratings (Subplot 1)

**Kode:**
```python
plt.scatter(actuals[:1000], preds[:1000], alpha=0.6)
plt.plot([0.5, 5], [0.5, 5], 'r--')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predicted vs Actual Ratings')
plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan scatter plot untuk membandingkan 1.000 rating aktual dengan prediksi, dilengkapi garis diagonal merah ideal.

**Parameter:**
- Transparansi: `alpha=0.6`
- Grid: `alpha=0.3`

**Insight:** Titik mendekati garis diagonal menunjukkan korelasi positif dengan error kecil (RMSE: 0.8395, MAE: 0.6423). Penyimpangan pada rating <2.0 menunjukkan area yang perlu perbaikan.

#### 2. Residuals Plot (Subplot 2)

**Kode:**
```python
residuals = np.array(preds) - np.array(actuals)
plt.scatter(preds[:1000], residuals[:1000], alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan scatter plot residual (selisih prediksi-aktual) untuk 1.000 data dengan garis nol merah.

**Parameter:**
- Transparansi: `alpha=0.6`
- Grid: `alpha=0.3`

**Insight:** Residual tersebar acak di sekitar nol tanpa bias. Outlier (>1 atau <-1) pada prediksi 3.0-4.0 menunjukkan potensi optimasi.

#### 3. Distribution of Residuals (Subplot 3)

**Kode:**
```python
plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan histogram distribusi residual dengan 50 bins.

**Parameter:**
- Warna: `skyblue`
- Transparansi: `alpha=0.7`
- Garis tepi: `black`
- Grid: `alpha=0.3`

**Insight:** Distribusi mendekati normal dengan puncak di 0. Lebar -1.5 hingga 1.5 konsisten dengan RMSE (0.8395) dan MAE (0.6423), dengan ekor menunjukkan error besar.

#### 4. Genre Distribution in Recommendations (Subplot 4)

**Kode:**
```python
sample_recs = get_content_recommendations('Forrest Gump (1994)', n_recommendations=20)
if isinstance(sample_recs, pd.DataFrame):
    all_genres_recs = sample_recs['genres'].str.split('|').explode()
    genre_counts_recs = all_genres_recs.value_counts().head(8)
    genre_counts_recs.plot(kind='bar', color='lightcoral', alpha=0.7)
    plt.title('Genre Distribution in Recommendations')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan bar plot distribusi 8 genre teratas dari 20 rekomendasi untuk "Forrest Gump (1994)".

**Parameter:**
- Warna: `lightcoral`
- Transparansi: `alpha=0.7`
- Rotasi label: 45 derajat
- Grid: `alpha=0.3`

**Insight:** Drama dan Romance dominan, sesuai genre "Forrest Gump" (Drama|Romance|War). War kurang terwakili, menunjukkan perlunya fitur tambahan. Bar plot genre rekomendasi mendukung relevansi untuk input spesifik.

#### 5. Actual vs Predicted Rating Distribution (Subplot 5)

**Kode:**
```python
plt.hist(actuals, bins=20, alpha=0.7, label='Actual', color='blue', edgecolor='black')
plt.hist(preds, bins=20, alpha=0.7, label='Predicted', color='red', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Actual vs Predicted Rating Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan histogram distribusi rating aktual dan prediksi dengan 20 bins.

**Parameter:**
- Warna: `blue` untuk aktual, `red` untuk prediksi
- Transparansi: `alpha=0.7`
- Garis tepi: `black`
- Legenda dan grid: `alpha=0.3`

**Insight:** Distribusi prediksi mirip aktual (puncak 3.5-4.0), tetapi lebih terkonsentrasi. Model bersifat konservatif untuk rating ekstrem (<2.0 atau >4.5).

#### 6. Model Performance Metrics (Subplot 6)

**Kode:**
```python
metrics = ['RMSE', 'MAE']
values = [rmse_final, mae_final]
colors = ['red', 'blue']
bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
plt.title('Model Performance Metrics')
plt.ylabel('Error Value')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.grid(True, alpha=0.3)
```

**Tujuan:** Menampilkan bar plot untuk RMSE (0.8395) dan MAE (0.6423).

**Parameter:**
- Warna: `red` untuk RMSE, `blue` untuk MAE
- Transparansi: `alpha=0.7`
- Garis tepi: `black`
- Anotasi nilai dan grid: `alpha=0.3`

**Insight:** RMSE (0.8395) > MAE (0.6423) karena sensitivitas terhadap error besar. Error <1.0 menunjukkan performa baik.

### Kesimpulan Visualisasi

#### Performa Model SVD
- **RMSE**: 0.8395
- **MAE**: 0.6423
- Residual terdistribusi normal dengan korelasi positif yang baik

### Content-Based Filtering
- Rekomendasi relevan untuk genre utama (Drama, Romance)
- **Area Perbaikan**: Genre sekunder (War) dan rating ekstrem perlu optimasi lebih lanjut

![Visualization](https://imgur.com/MKY5sBC.png "Visualization")
**Gambar 3**: Visualization

## Kesimpulan
- Sistem rekomendasi hybrid telah dikembangkan dengan menggabungkan content-based filtering (berbasis genre dan tag) serta collaborative filtering (menggunakan Surprise SVD).
- Evaluasi collaborative filtering menunjukkan hasil prediksi yang cukup akurat dengan RMSE ~0.84 dan MAE ~0.64.
- Content-based filtering mampu memberikan rekomendasi relevan untuk beberapa film (misalnya Toy Story (1995)), namun tidak konsisten untuk semua film (misalnya The Matrix (1999)).
- Evaluasi hybrid system pada beberapa pengguna menunjukkan Precision@10 = 0, mengindikasikan bahwa integrasi metode belum sepenuhnya optimal atau bahwa user tersebut mungkin belum memiliki data yang cukup.
- Visualisasi seperti heatmap dan metrik evaluasi membantu dalam memahami kualitas dan batasan masing-masing pendekatan.

## Saran
1. Menambahkan fitur konten yang lebih kaya seperti plot summary, aktor, atau sutradara untuk memperkuat content-based filtering.
2. Menerapkan algoritma lanjutan seperti Alternating Least Squares (ALS) atau Neural Collaborative Filtering untuk meningkatkan performa collaborative filtering.
3. Mengatasi masalah cold-start dengan pendekatan berbasis popularitas atau metadata untuk item dan pengguna baru.
4. Melakukan hyperparameter tuning lebih lanjut, seperti jumlah faktor pada SVD, learning rate, atau max_features pada TF-IDF.
5. Menggunakan metrik tambahan seperti NDCG@k atau MAP@k untuk evaluasi yang lebih menyeluruh terhadap relevansi rekomendasi.

## Referensi
1. GroupLens. (n.d.). MovieLens Dataset. https://grouplens.org/datasets/movielens/
2. Ricci, F., Rokach, L., & Shapira, B. (2011). *Recommender Systems Handbook*. Springer.  
3. Surprise Documentation. http://surpriselib.com/
4. Scikit-learn Documentation. https://scikit-learn.org/stable/ 
5. Falk, K. (2019). *Practical Recommender Systems*. Manning Publications.
 
