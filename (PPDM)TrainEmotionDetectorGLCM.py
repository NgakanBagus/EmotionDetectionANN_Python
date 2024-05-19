# import required packages
import cv2
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten
from keras.src.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import tensorflow

# Initialize image data generator
train_data_gen = ImageDataGenerator()
validation_data_gen = ImageDataGenerator()

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
        'archive/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

glcm_features_train = []
labels_train = []
# Iterate over each batch of the dataset
for images_batch, labels_batch in train_generator:
        images_batch = (images_batch * 255).astype(np.uint8)

        for image, label in zip(images_batch, labels_batch):  # Use zip to iterate over images and labels simultaneously
                # Make the 3D array image outputted from 'flow_from_directory' into a 2D array
                img = image.squeeze()
                
                # Compute GLCM
                glcm = graycomatrix(img, distances=[1], angles=[0])
                contrast = graycoprops(glcm, 'contrast').ravel()
                dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
                correlation = graycoprops(glcm, 'correlation').ravel()
                energy = graycoprops(glcm, 'energy').ravel()
                homogeneity = graycoprops(glcm, 'homogeneity').ravel()
                
                # Concatenate GLCM features
                glcm_features = np.concatenate((contrast, dissimilarity, correlation, energy, homogeneity))
                
                # Append GLCM features and labels
                glcm_features_train.append(glcm_features)
                labels_train.append(label)

        # Check if all images in the dataset have been processed
        if len(glcm_features_train) >= len(train_generator.filenames):
                break

glcm_features_train = np.array(glcm_features_train)
glcmTrainFlat = glcm_features_train.reshape((3191, 5))
labels_train = np.array(labels_train)

print(len(glcm_features_train))
print(len(labels_train))

# Preprocess all validation images
# Rest is the same as the training data
validation_generator = validation_data_gen.flow_from_directory(
        'archive/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

glcm_features_val = []
labels_val = []

for images_batch, labels_batch in validation_generator:
        images_batch = (images_batch * 255).astype(np.uint8)

        for image, label in zip(images_batch, labels_batch):  # Use zip to iterate over images and labels simultaneously
                # Make the 3D array image outputted from 'flow_from_directory' into a 2D array
                img = image.squeeze()
                
                # Compute GLCM
                glcm = graycomatrix(img, distances=[1], angles=[0])
                contrast = graycoprops(glcm, 'contrast').ravel()
                dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
                correlation = graycoprops(glcm, 'correlation').ravel()
                energy = graycoprops(glcm, 'energy').ravel()
                homogeneity = graycoprops(glcm, 'homogeneity').ravel()
                
                # Concatenate GLCM features
                glcm_features = np.concatenate((contrast, dissimilarity, correlation, energy, homogeneity))
                
                # Append GLCM features and labels
                glcm_features_val.append(glcm_features)
                labels_val.append(label)

        if len(glcm_features_val) >= len(validation_generator.filenames):
                break

glcm_features_val = np.array(glcm_features_val)
glcmValFlat = glcm_features_val.reshape((809, 5))
labels_val = np.array(labels_val)

print(len(glcm_features_val))
print(len(labels_val))

# create model structure
emotion_model = Sequential()

emotion_model.add(Dense(2048, input_shape=(5, ), activation='relu'))
emotion_model.add(Dropout(0.1))
emotion_model.add(Dense(2048, activation='relu'))
emotion_model.add(Dropout(0,15))
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.1))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the neural network/model
logdir='logs'
tensorboar_callback=tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)

hist = emotion_model.fit(
        x=glcmTrainFlat,
        y=labels_train,
        epochs=40,
        batch_size=64,
        validation_data=(glcmValFlat, labels_val),
        validation_batch_size=64,
        callbacks=[tensorboar_callback])

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.weights.h5')

emotion_model.summary()

plt.plot(hist.history['accuracy'], color='green')
plt.plot(hist.history['val_accuracy'], color='yellow')
plt.title("acc(g) --- val_acc(y)")
plt.show()

plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.title("loss(r) --- val_loss(b)")
plt.show()