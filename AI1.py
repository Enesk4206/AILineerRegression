# Gerekli kütüphaneleri yükle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Basit bir veri seti oluşturalım
X = np.array([[1], [2], [3], [4], [5], [6]])    #6 satır 1 sütun
y = np.array([1, 2, 3, 4, 5, 6])                #1 satır 6 sütun

# X_train: Eğitim için kullanılan girdi verileri. 80
# X_test: Test için kullanılan girdi verileri. 20
# y_train: Eğitim için kullanılan hedef verileri. 80 
# y_test: Test için kullanılan hedef verileri. 20
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
# X_train: Eğitim verileri (özellikler).
# y_train: Eğitim verilerine karşılık gelen hedef değerler.
model.fit(X_train, y_train)  #liner regrasyon ile 

# Test setinde tahmin yap
y_pred = model.predict(X_test)

# Sonuçları görselleştir
plt.scatter(X_test, y_test, color='red')  # Gerçek değerler
plt.plot(X_test, y_pred, color='blue')    # Tahmin edilen değerler
plt.title('Lineer Regresyon Tahmin')
plt.xlabel('X Değeri')
plt.ylabel('Y Değeri')
plt.show()
