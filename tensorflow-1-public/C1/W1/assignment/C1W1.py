import tensorflow as tf
import numpy as np


def house_model():
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([1, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mse')

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    ### END CODE HERE
    return model

# Get your trained model
model = house_model()

new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)