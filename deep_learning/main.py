import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

length_of_signal = 100
number_of_seeds = 10

X, y = pickle.load(open('data.obj', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Sequential()
model.add(Flatten(input_shape=(length_of_signal, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(number_of_seeds, activation='relu'))

model.compile(loss='mae',
              optimizer='adam')

model.fit(X_train, y_train, epochs=500, validation_split=0.2)

prediction = model.predict(X_test)
number_of_samples = len(prediction)
real_prediction = np.zeros((number_of_samples, length_of_signal, 1))
scaler = MinMaxScaler()

for sample in range(number_of_samples):
    for seed in range(number_of_seeds):
        np.random.seed(seed)

        new_addition = np.random.rand(length_of_signal, 1)
        for element in range(len(new_addition)):
            new_addition[element] = new_addition[element] * prediction[sample][seed]

        real_prediction[sample] += new_addition
        real_prediction[sample] = scaler.fit_transform(real_prediction[sample])

plt.scatter(range(length_of_signal), X_test[0])
plt.scatter(range(length_of_signal), real_prediction[0])
plt.show()

error = np.mean(np.abs((X_test[sample] - real_prediction[sample])))
print(error)
