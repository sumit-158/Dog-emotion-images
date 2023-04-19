import tensorflow as tf
import os
import numpy

images = os.path.join("images")
os.listdir(images)

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = images,
    seed = 32,
    validation_split = 0.2,
    image_size = (60,60),
    subset="training"
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    directory = images,
    seed= 32,
    validation_split=0.2,
    image_size=(60,60),
    subset = "validation"
)
optimization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = raw_train_ds.map(lambda x,y : (optimization_layer(x),y))
val_ds = raw_val_ds.map(lambda x,y : (optimization_layer(x),y))
model = tf.keras.models.Sequential([
    #first_layer
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),

    #second_layer
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),

    #third_layer 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(5,activation="softmax")
])
print("model processing done!")
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimization = tf.keras.optimizers.Adam()
model.compile(
    optimizer=optimization,
    loss=loss,
    metrics=["accuracy"]
)
history = model.fit(train_ds,epochs=20)
print("model traning done!")
model.evaluate(val_ds,verbose=10)
model.save("Dog_emotions.h5")
