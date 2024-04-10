import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare test data
df = pd.read_csv('churn.csv')
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Prepare model
model = keras.Sequential()
model.add(layers.Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

result = accuracy_score(y_test, y_hat)
print("Accuracy: " + str(result))