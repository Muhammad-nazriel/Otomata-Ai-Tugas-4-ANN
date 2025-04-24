import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

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
history = model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

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
    'Prediksi Produksi 2025 (Kuintal)': predictions.flatten()
})

# Tampilkan hasil prediksi
print("\nPrediksi Produksi Buah dan Sayur Tahun 2025:")
print(results_df.to_string(index=False))

# Plot history training
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualisasi hasil prediksi
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Komoditi', y='Prediksi Produksi 2025 (Kuintal)')
plt.xticks(rotation=45, ha='right')
plt.title('Prediksi Produksi Buah dan Sayur Tahun 2025')
plt.tight_layout()
plt.show()
