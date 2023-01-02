import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

PATH = "F:\\the_face_project"
files = os.listdir(PATH)

model = keras.Sequential()

model.add(keras.layers.Dense(2048, input_shape=((92* 72*3,)), activation="relu"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(1024, activation="relu"))
model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Dense(350, activation="relu"))
#model.add(keras.layers.BatchNormalization())
#
#model.add(keras.layers.Dense(732, activation="relu"))
model.add(keras.layers.Dense(2048, activation="relu"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(92*72*3, activation="relu"))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=opt)

model.summary()

ITER = 12

loss = []
val_loss = []

for it in range(ITER):

    for file in files:

        if file == "img_align_celeba":
            continue

        x = np.load(PATH + "\\" + file)
        x_ = x.reshape((len(x), -1))
        SPLIT = int(len(x)*0.9)

        hist = model.fit(x_[0:SPLIT], x_[0:SPLIT], batch_size=200, epochs=1, validation_data=(x_[SPLIT:-1], x_[SPLIT:-1]))

        loss = [*loss, *hist.history["loss"]]
        val_loss = [*val_loss, *hist.history["val_loss"]]


    model.save("auto2.h5")

    plt.plot(loss)
    plt.plot(val_loss)
    plt.yscale("log")
    plt.savefig("train_loss.png")

