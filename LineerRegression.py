import numpy as np
from sklearn.linear_model import LinearRegression

# Ürünlerin fiyatları (X = Ürün numarası, y = Fiyat)
X = np.array([[1], [2], [3], [4]])  # Ürün A, B, C, D
y = np.array([250, 100, 300, 150])  # Karmaşık fiyatlar

# Model oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Ürün E'nin fiyatını tahmin et (X = 5)
predicted_price = model.predict(np.array([[5]]))
print(f"Ürün E'nin tahmin edilen fiyatı: {predicted_price[0]:.2f} ₺")
