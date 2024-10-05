import numpy as np
from sklearn.linear_model import LinearRegression

# products priceses (X = number of products, y = Price)
X = np.array([[1], [2], [3], [4],[5], [6], [7], [8]])  # Ürün A, B, C, D ....
y = np.array([250, 100, 300, 150 , 200 , 500 , 1110, 2000])  # Karmaşık fiyatlar

# Model oluştur ve eğit
model = LinearRegression() # defined certain coeffiecint(slope) * number of products + intercept Y axios top value  e.g(50* number of product + 100)



model.fit(X, y) # represents:  X are independent values  y are dependant values 
# model fit train to models


# for product 9 train new predict  (X = 9)
predicted_price = model.predict(np.array([[9]]))
print(f"Ürün E'nin tahmin edilen fiyatı: {predicted_price[0]:.2f} ₺")
