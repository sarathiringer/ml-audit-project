import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def train():
    data = pd.read_csv('data/adult_preprocessed.csv', index_col=0)

    X = np.array(data[data.columns[:-1]])
    y = np.array([1 if x == '>50K' else 0 for x in data['salary']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = keras.Sequential()
    model.add(layers.Dense(32, activation="relu", input_shape=(12,)))
    model.add(layers.Dense(64, activation="relu", input_shape=(32,)))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
    print('Model trained')

    # Save the model
    model.save('model/salary_model')
    print('Model saved')
