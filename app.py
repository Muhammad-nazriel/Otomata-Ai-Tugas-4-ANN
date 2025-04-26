from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

@app.route('/')
def home():
    # Baca dataset
    df = pd.read_csv('distanhor-od_17906_prdks_buah_buahan_sayuran_thnan_bst__komoditi_v1_data.csv')

    # Label encoding untuk kolom komoditi
    le = LabelEncoder()
    df['komoditi_encoded'] = le.fit_transform(df['komoditi'])

    # Persiapkan data untuk model
    X = df[['tahun', 'komoditi_encoded']].values
    y = df['jumlah_produksi'].values

    # Normalisasi input menggunakan MinMaxScaler
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Normalisasi output
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Bangun model ANN
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Training model
    model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    # Buat data untuk prediksi tahun 2025
    komoditi_list = df['komoditi'].unique()
    pred_data = []
    for komoditi in komoditi_list:
        komoditi_encoded = le.transform([komoditi])[0]
        pred_data.append([2025, komoditi_encoded])

    pred_data = np.array(pred_data)
    pred_data_scaled = scaler_X.transform(pred_data)

    # Lakukan prediksi
    predictions_scaled = model.predict(pred_data_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Buat dataframe hasil prediksi
    results_df = pd.DataFrame({
        'Komoditi': komoditi_list,
        'Prediksi_Produksi_2025': predictions.flatten()
    })

    # Sort predictions by value descending
    results_df = results_df.sort_values('Prediksi_Produksi_2025', ascending=False)

    # Format predictions to readable numbers
    results_df['Prediksi_Produksi_2025'] = results_df['Prediksi_Produksi_2025'].map('{:,.0f}'.format)

    # Get historical data for comparison
    historical_data = df.groupby('komoditi')['jumlah_produksi'].agg(['mean', 'min', 'max']).round(2)
    historical_data = historical_data.reset_index()
    
    return render_template('index.html', 
                         predictions=results_df.to_dict('records'),
                         historical_data=historical_data.to_dict('records'))

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/data')
def data():
    df = pd.read_csv('distanhor-od_17906_prdks_buah_buahan_sayuran_thnan_bst__komoditi_v1_data.csv')
    data = df.to_dict('records')
    return render_template('data.html', data=data, title='Historical Data')

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
