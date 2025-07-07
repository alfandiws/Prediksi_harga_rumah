from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset referensi dan encoder
df = pd.read_csv('df_model.csv')
location_map = dict(df[['listing-location', 'location_encoded']].drop_duplicates().values.tolist())

# Ambil harga tanah rata-rata dari dataset
harga_tanah_per_m = df['listing-floorarea 2'].mean()

# Load model dan label encoder
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template(
        'index.html',
        price_prediction=0,
        location_list=location_map.keys()
    )

@app.route('/predict', methods=['POST'])
def predict():
    location_name = request.form['listing_location']
    location_encoded = location_map[location_name]

    bed = int(request.form['bed'])
    bath = int(request.form['bath'])
    floorarea = float(request.form['listing-floorarea'])

    features = [location_encoded, bed, bath, floorarea, harga_tanah_per_m]
    prediction = model.predict([features])[0]
    output = int(round(prediction))
    price_formatted = f"{output:,}".replace(",", ".")

    return render_template(
        'index.html',
        price_prediction=price_formatted,
        location_list=location_map.keys(),
        location_name=location_name,
        bed=bed,
        bath=bath,
        listing_floorarea=floorarea
    )

if __name__ == '__main__':
    app.run(debug=True)
