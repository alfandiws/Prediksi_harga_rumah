import streamlit as st
import pandas as pd
import joblib

#Load Model & Dataset
df = pd.read_csv('df_model.csv')
model = joblib.load('random_forest_model.pkl')

#Lokasi & Encoding
location_map = dict(df[['listing-location', 'location_encoded']].drop_duplicates().values.tolist())
location_list = list(location_map.keys())

harga_tanah_per_m = df['listing-floorarea 2'].mean()

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.markdown("""
    <div style="background-color:#2563EB;padding:20px;border-radius:10px;text-align:center">
        <h1 style="color:white;font-size:40px;">Prediksi Harga Rumah</h1>
    </div>
""", unsafe_allow_html=True)

st.write("")  
col1, col2 = st.columns([1, 1])

# Form Input di kolom kiri
with col1:
    st.subheader("Form Prediksi Harga")

    lokasi = st.selectbox("Lokasi", location_list)
    bed = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, step=1)
    bath = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, step=1)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10.0, step=1.0)

    predict_button = st.button("🔍 Prediksi Sekarang")

# Output Hasil Prediksi di kolom kanan 
with col2:
    st.subheader("Estimasi Harga Rumah")

    if predict_button:
        # Ambil nilai encoded dari lokasi
        lokasi_encoded = location_map[lokasi]

        # Susun fitur sesuai urutan training
        features = [[
            lokasi_encoded,    
            bed,               
            bath,              
            luas_bangunan,     
            harga_tanah_per_m  
        ]]

        # Prediksi harga
        prediction = model.predict(features)[0]
        output = int(round(prediction))
        price_formatted = f"Rp {output:,}".replace(",", ".")

        # Tampilkan hasil
        st.markdown(f"""
        <h2 style="color:#2563EB;text-align:center;font-size:30px;">
            {price_formatted}
        </h2>
        """, unsafe_allow_html=True)

        # Tampilkan ringkasan input
        st.markdown("---")
        st.markdown("### Detail Input:")
        st.write(f" Lokasi: **{lokasi}**")
        st.write(f" Kamar Tidur: **{bed}**")
        st.write(f" Kamar Mandi: **{bath}**")
        st.write(f" Luas Bangunan: **{luas_bangunan} m²**")
    else:
        st.info("Isi form di sebelah kiri dan klik **Prediksi Sekarang** untuk melihat estimasi harga.")
