import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pengaturan Tema
st.set_page_config(
    page_title="Prediksi Poin & Peluang Juara EPL",
    layout="wide",
    page_icon="⚽"
)

# Judul Aplikasi
st.title("⚽ Prediksi Poin Akhir dan Peluang Juara Liga Inggris")
st.markdown("Masukkan data performa tim untuk memprediksi **poin akhir** dan **peluang menjadi juara**.")
st.divider()

# Sidebar untuk Input Data
st.sidebar.title("⚙️ Input Data Tim")
team_name = st.sidebar.text_input("Nama Tim", placeholder="Contoh: Manchester United")
W = st.sidebar.number_input("Jumlah Kemenangan (W)", min_value=0, max_value=38, step=1)
D = st.sidebar.number_input("Jumlah Seri (D)", min_value=0, max_value=38, step=1)
L = st.sidebar.number_input("Jumlah Kalah (L)", min_value=0, max_value=38, step=1)
GF = st.sidebar.number_input("Gol Memasukkan (GF)", min_value=0, step=1)
GA = st.sidebar.number_input("Gol Kemasukan (GA)", min_value=0, step=1)

# Validasi Data
total_matches = W + D + L
if total_matches != 38:
    st.sidebar.error(f"Jumlah total pertandingan harus 38. Saat ini: {total_matches}.")
GD = GF - GA
st.sidebar.markdown(f"**Selisih Gol (GD):** {GD}")

# Load Model dan Data
try:
    with open('epl_forecasting_model.pkl', 'rb') as file:
        model = pickle.load(file)
    standings = pd.read_csv('EPL_Standings_final_2024.csv')

    # Kolom Utama untuk Data dan Grafik
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Data Historis Poin Juara")
        champions_points = standings[standings['Pos'] == 1]['Pts']
        historical_champion_points = champions_points.values

        st.dataframe(champions_points.reset_index(drop=True), height=200)

        st.subheader("Grafik Distribusi Perolehan Poin Juara Liga Inggris")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            historical_champion_points,
            kde=True, bins=10, ax=ax,
            color=sns.color_palette("coolwarm", as_cmap=True)(0.5)
        )
        ax.axvline(np.mean(historical_champion_points), color='red', linestyle='--', label='Rata-rata Poin')
        ax.set_title("Distribusi Perolehan Poin Juara Liga Inggris (Historis)", fontsize=14)
        ax.set_xlabel("Poin", fontsize=12)
        ax.set_ylabel("Frekuensi", fontsize=12)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Tren Total Poin Juara per Musim")
        seasons = standings[standings['Pos'] == 1]['Season']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(seasons, historical_champion_points, marker='o', color='green', label='Poin Juara')
        ax.axhline(np.mean(historical_champion_points), color='red', linestyle='--', label='Rata-rata Poin')
        ax.set_title("Tren Poin Juara per Musim", fontsize=14)
        ax.set_xlabel("Musim", fontsize=12)
        ax.set_ylabel("Poin", fontsize=12)
        ax.set_xticks(seasons)
        ax.set_xticklabels(seasons, rotation=45, fontsize=10)
        ax.legend()
        st.pyplot(fig)

    # Prediksi Poin dan Peluang
    st.subheader("🔮 Prediksi Poin Akhir dan Peluang Juara")
    if st.button("Prediksi"):
        if total_matches == 38:
            input_data = pd.DataFrame({
                "W": [W],
                "D": [D],
                "L": [L],
                "GF": [GF],
                "GA": [GA],
                "GD": [GD],
            })

            predicted_points = model.predict(input_data)[0]

            # Hitung peluang
            avg_champion_points = np.mean(historical_champion_points)
            std_champion_points = np.std(historical_champion_points)
            z_score = (predicted_points - avg_champion_points) / std_champion_points
            championship_probability = 1 / (1 + np.exp(-z_score)) * 100

            st.success(f"Tim: {team_name if team_name else 'Tidak disebutkan'}")
            st.success(f"Prediksi Poin Akhir: {predicted_points:.2f}")
            st.info(f"Peluang Menjadi Juara: {championship_probability:.2f}%")
        else:
            st.error("Jumlah total pertandingan harus 38!")

except FileNotFoundError:
    st.error("File CSV atau model tidak ditemukan! Harap unggah file.")
