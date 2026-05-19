# Dashboard Pemetaan Risiko Anemia Ibu Hamil

## Ringkasan Proyek
Dashboard Streamlit ini dirancang untuk mendukung penelitian anemia pada ibu hamil di Kabupaten Lombok Utara. Aplikasi menggabungkan pemetaan hotspot spasial, analitik text mining terhadap kendala konsumsi Tablet Tambah Darah (TTD), dan segmentasi risiko berbasis machine learning untuk menyusun rekomendasi intervensi yang dapat ditindaklanjuti oleh tenaga kesehatan.

## Fitur Utama
- **Dashboard Utama** dengan metrik kunci, visualisasi ringkas, dan akses cepat ke modul analitik.
- **Modul 1 – Peta Hotspot Anemia**: Clustering K-Means spasial dengan peta interaktif Folium, statistik cluster, ekspor hasil.
- **Modul 2 – Analisis Kendala TTD**: NLP berbahasa Indonesia (tokenisasi, stemming Sastrawi, sentimen), word cloud, heatmap geografis, rekomendasi otomatis.
- **Modul 3 – Segmentasi Risiko**: Random Forest classifier dengan tuning hyperparameter, evaluasi lengkap, formulir prediksi kasus baru, batch prediction, analisis what-if, pencarian kasus serupa, rekomendasi intervensi.
- **Tema profesional** dengan styling kustom, kompatibel untuk deployment produksi.

## Struktur Proyek
```
anemia-dashboard/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── custom.css
│   └── images/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── deployment_guide.md
│   └── user_manual.md
├── models/
├── modules/
│   ├── modul1_hotspot_mapping.py
│   ├── modul2_ttd_barriers.py
│   └── modul3_risk_segmentation.py
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── visualization.py
└── assets/
    └── logo.png (opsional)
```

## Persyaratan Sistem
- Python 3.9 atau lebih baru
- Pip / virtual environment (direkomendasikan)
- Streamlit 1.28 atau lebih baru

## Instalasi
1. Klona repositori ini.
2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Letakkan file Excel survei pada folder `data/raw/` sesuai nama di `config.py`.

## Menjalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan terbuka pada browser lokal (default: http://localhost:8501).

## Panduan Penggunaan
1. **Dashboard Utama**: pantau metrik ringkas dan visualisasi awal. Gunakan tombol CTA untuk berpindah modul.
2. **Modul 1**: atur jumlah cluster dan fitur tambahan di sidebar, eksplor peta, unduh hasil.
3. **Modul 2**: gunakan filter demografis, telaah word cloud per kategori, unduh rekap analisis.
4. **Modul 3**: lihat performa model, lakukan prediksi kasus baru atau batch, simulasikan skenario, unduh daftar risiko tinggi.
5. **About**: baca metodologi, sumber data, dan kontak proyek.

## Screenshot (Placeholder)
- ![Dashboard Overview](docs/images/dashboard_overview.png)
- ![Hotspot Map](docs/images/hotspot_map.png)
- ![TTD Barriers](docs/images/ttd_barriers.png)
- ![Risk Segmentation](docs/images/risk_segmentation.png)

## Deployment
1. Siapkan VPS (Ubuntu) dengan Python 3.9+ dan pip.
2. Salin kode proyek ke server (git clone atau rsync).
3. Instal dependensi menggunakan `pip install -r requirements.txt`.
4. Gunakan `pm2`, `systemd`, atau `supervisor` untuk menjaga layanan Streamlit tetap berjalan.
5. Konfigurasikan reverse proxy (Nginx) untuk custom domain dan HTTPS.
6. Perbarui `.streamlit/config.toml` untuk konfigurasi server produksi.
7. Pastikan file data ditempatkan di `data/raw/` dan perbarui `config.py` bila nama/letak berbeda.

Panduan rinci tersedia pada `docs/deployment_guide.md`.

## Kontribusi
- Fork repositori
- Buat branch fitur: `git checkout -b fitur-anda`
- Commit perubahan: `git commit -m "Tambah fitur X"`
- Push branch: `git push origin fitur-anda`
- Ajukan pull request dengan deskripsi rinci

## Lisensi
Proyek ini dilisensikan di bawah MIT License. Silakan lihat file LICENSE jika tersedia.

## Kontak
- Email: prayaadhiganaglobal@gmail.com
- WhatsApp: +62 812-3456-7890
- Website: https://kluanemia.example.com

---
Dikembangkan oleh tim penelitian KLU untuk meningkatkan intervensi anemia pada ibu hamil melalui pemanfaatan analitik modern.
