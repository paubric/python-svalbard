import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle


def generate_data(length_of_signal, number_of_seeds, number_of_samples):
    scaler = MinMaxScaler()

    X = np.zeros((number_of_samples, length_of_signal, 1))
    y = np.zeros((number_of_samples, number_of_seeds))

    for sample in range(number_of_samples):
        np.random.seed()
        y[sample] = np.random.rand(number_of_seeds)

        for seed in range(number_of_seeds):
            np.random.seed(seed)

            new_addition = np.random.rand(length_of_signal, 1)
            for element in range(len(new_addition)):
                new_addition[element] = new_addition[element] * y[sample][seed]

            X[sample] += new_addition
            X[sample] = scaler.fit_transform(X[sample])
    
    return X, y

data = generate_data(100, 10, 10000)
pickle.dump(data, open('data.obj', 'wb'))