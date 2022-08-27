import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import valohai


def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])

def main():
    input_path = valohai.inputs('dataset').path()
    data = pd.read_csv(input_path, index_col=0)

    X = np.array(data[data.columns[:-1]])
    y = np.array([1 if x == '>50K' else 0 for x in data['salary']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation="relu", input_shape=(12,)))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=(32,)))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    model.fit(
        X_train, 
        y_train, 
        batch_size=valohai.parameters('batch_size').value, 
        epochs=valohai.parameters('epochs').value, 
        validation_split=valohai.parameters('validation_split').value,
        callbacks=[callback]
    )
    print('Model trained')

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test,  y_test, verbose=2)
    with valohai.logger() as logger:
        logger.log('test_accuracy', test_accuracy)
        logger.log('test_loss', test_loss)
    print('Model evaluated')

    # Save the model
    path = valohai.outputs().path('salary_model')
    model.save(path)
    print('Model saved')


if __name__ == '__main__':
    main()