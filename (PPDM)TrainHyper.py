# import required packages
import cv2
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten
from keras.src.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'archive/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'archive/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotion_model = Sequential()

emotion_model.add(Flatten(input_shape=(48,48,1)))
emotion_model.add(Dense(2048, activation='relu'))
emotion_model.add(Dropout(0.2))
emotion_model.add(Dense(2048, activation='relu'))
emotion_model.add(Dropout(0,15))
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0,1))
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the neural network/model
logdir='logs'
tensorboar_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)

hist = emotion_model.fit(
        train_generator,
        epochs=40,
        batch_size=64,
        validation_data=validation_generator,
        validation_batch_size=64,
        callbacks=[tensorboar_callback])

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.weights.h5')


plt.plot(hist.history['loss'], label='loss', color='red')
plt.plot(hist.history['val_loss'], label='val_loss', color='blue')
plt.show()

plt.plot(hist.history['accuracy'], label='accuracy', color='green')
plt.plot(hist.history['val_accuracy'], label='val_accuracy', color='yellow')
plt.show()