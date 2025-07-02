# Impor library Flask untuk web app, pandas untuk manipulasi data, numpy, joblib, dan keras untuk model ML
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model LSTM yang sudah dilatih sebelumnya tanpa kompilasi ulang
model = load_model("model_lstm.h5", compile=False)

# Load scaler (normalisasi) yang digunakan saat training
scaler = joblib.load("scaler.save")

# Load dataset dari file Excel
df = pd.read_excel("rumah_tangga.xlsx")

# Filter data hanya untuk perempuan
df = df[df['Jenis Kelamin'] == 'Perempuan']

# Ambil list unik daerah dan kelompok umur untuk pilihan input form
daerah_list = sorted(df['Daerah'].unique())
umur_list = sorted(df['Kelompok Umur'].unique())

# Route utama, mendukung metode GET dan POST
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ambil input dari form
        daerah_input = request.form["daerah"]
        umur_input = request.form["umur"]
        tahun_input = int(request.form["tahun"])

        # Filter data berdasarkan input pengguna
        df_filtered = df[
            (df["Daerah"] == daerah_input) &
            (df["Kelompok Umur"] == umur_input) &
            (df["Tahun"] <= tahun_input)
        ].sort_values(by="Tahun")

        # Ambil 5 data terakhir sebagai input ke LSTM
        seq_len = 5
        df_seq = df_filtered.tail(seq_len)

        # Jika data kurang dari 5 tahun, tampilkan pesan error
        if len(df_seq) < seq_len:
            return render_template("index.html",
                                   daerah_list=daerah_list,
                                   umur_list=umur_list,
                                   prediction=False,
                                   error="Data tidak cukup untuk prediksi (butuh 5 tahun terakhir).")

        # One-hot encoding untuk data training dan input
        df_all = pd.get_dummies(df, columns=["Daerah", "Kelompok Umur"])
        df_seq_enc = pd.get_dummies(df_seq, columns=["Daerah", "Kelompok Umur"])

        # Tambahkan kolom yang hilang di df_seq_enc agar konsisten dengan df_all
        for col in df_all.columns:
            if col not in df_seq_enc.columns:
                df_seq_enc[col] = 0  # isi dengan 0 jika kolom tidak ada

        # Tentukan kolom fitur (kecuali target dan non-fitur)
        feature_cols = [col for col in df_all.columns if col not in ['Tahun', 'Jenis Kelamin',
                                                                     'Belum Kawin', 'Kawin',
                                                                     'Cerai Hidup', 'Cerai Mati']]
        # Susun ulang kolom agar sesuai urutan saat training
        df_seq_enc = df_seq_enc[feature_cols]

        # Ubah ke array numpy dan reshape ke bentuk (1, 5, fitur) sesuai input LSTM
        input_data = df_seq_enc.values.astype(np.float32).reshape(1, seq_len, len(feature_cols))

        # Lakukan prediksi menggunakan model
        prediction = model.predict(input_data)[0]

        # Inverse scaling untuk mendapatkan nilai asli
        hasil = scaler.inverse_transform([prediction])[0]

        # Pisahkan hasil prediksi ke dalam kategori
        persen_belum_kawin = hasil[0]
        persen_kawin = hasil[1]
        persen_cerai_hidup = hasil[2]
        persen_cerai_mati = hasil[3]

        # Asumsikan populasi 10.000 orang untuk simulasi jumlah
        total_populasi = 10000
        jumlah_belum_kawin = round((persen_belum_kawin / 100) * total_populasi)
        jumlah_kawin = round((persen_kawin / 100) * total_populasi)
        jumlah_cerai_hidup = round((persen_cerai_hidup / 100) * total_populasi)
        jumlah_cerai_mati = round((persen_cerai_mati / 100) * total_populasi)

        # Tampilkan hasil prediksi ke template HTML
        return render_template("index.html",
                               daerah_list=daerah_list,
                               umur_list=umur_list,
                               prediction=True,
                               tahun=tahun_input,
                               persen_belum_kawin=persen_belum_kawin,
                               persen_kawin=persen_kawin,
                               persen_cerai_hidup=persen_cerai_hidup,
                               persen_cerai_mati=persen_cerai_mati,
                               jumlah_belum_kawin=jumlah_belum_kawin,
                               jumlah_kawin=jumlah_kawin,
                               jumlah_cerai_hidup=jumlah_cerai_hidup,
                               jumlah_cerai_mati=jumlah_cerai_mati)

    # Untuk request GET (awal), tampilkan form input saja
    return render_template("index.html",
                           daerah_list=daerah_list,
                           umur_list=umur_list,
                           prediction=False)

# Jalankan app Flask dalam mode debug
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

