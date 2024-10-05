import numpy as np
from sklearn.linear_model import LinearRegression

# E.g(A store wants to predict the price of a new product brand)

#define X: size of products 1:small, 2: medium, 3: large
#define Y: price of products
X = np.array([[1],[2],[3],[2],[1],[3],[2],[2],[1]])
y = np.array([300,33,222,100,200,300,500,400,1000])

model = LinearRegression()
model.fit(X ,y)

#new product size
new_product_size = np.array([[3]])
predict_price = model.predict(new_product_size)

print(f'The product predict price: {predict_price}:.2f ') #194.318181
 