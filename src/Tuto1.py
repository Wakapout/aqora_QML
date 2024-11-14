import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pennylane as qml
import tensorflow as tf

# QUANTUM LAYER
dev = qml.device("default.qubit", wires=3)

nb_qubit = 3

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs,wires=range(nb_qubit))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(nb_qubit))
    return [qml.expval(qml.PauliY(wires=i)) for i in range(nb_qubit)]


import matplotlib.pyplot as plt

n_layer = 3
weights_shape = {"weights": (n_layer,nb_qubit,3) }
qlayer = qml.qnn.KerasLayer(qnode,weights_shape,output_dim=nb_qubit)

clayer_1 = tf.keras.layers.Dense(2)
clayer_2 = tf.keras.layers.Dense(2, activation="softmax")

model = tf.keras.models.Sequential([clayer_1,qlayer,clayer_2])

opt = tf.keras.optimizers.SGD(learning_rate=0.2)
model.compile(opt,loss='mae', metrics=['accuracy'])

# TRAINING

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.1)
y_hot = tf.keras.utils.to_categorical(y, num_classes=2)

X = X.astype("float32")
y_hot = y_hot.astype("float32")
model.fit(X,y_hot, epochs=6, batch_size=5, validation_split=0.25)

# DISPLAY RESULTS
X_val, y_val = make_moons(n_samples=100, noise=0.1)
y_hot_pred = model.predict(X_val)
y_pred = tf.argmax(y_hot_pred, axis=1)
acc = tf.keras.metrics.Accuracy()
acc(y_pred, y_val)

c = ['#1f77b4' if y_ == 0 else '#ff7f0e' for y_ in y_pred]
plt.axis('off')
plt.scatter(X_val[:,0], X_val[:,1], c=c)
plt.show()