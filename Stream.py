import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import cv2
from streamlit_option_menu import option_menu
import threading
import pandas as pd

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('D:\PPDM\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:\PPDM\emotion_model.weights.h5")
print("Loaded model from disk")

def evaluate_model():
    global emotion_model
    global emotion_dict
    
    test_data_gen = ImageDataGenerator(rescale=1./255)

    # Preprocess test images
    test_generator = test_data_gen.flow_from_directory(
        'archive/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    # Do prediction on test data
    predictions = emotion_model.predict(test_generator)

    # Display confusion matrix
    st.write("Confusion Matrix:")
    predictions = emotion_model.predict(test_generator)
    c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
    st.write(c_matrix)

    # Visualize confusion matrix
    st.write("Visualized Confusion Matrix:")
    fig, ax = plt.subplots()
    cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # Classification report
    st.write("Hasil Akurasi:")
    report = classification_report(test_generator.classes, predictions.argmax(axis=1), output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

running = False
cap = cv2.VideoCapture(0)

def detect_emotion():
    global running
    global emotion_model
    global emotion_dict
    global cap

    # start the webcam feed
    while running:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('D:\PPDM\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def main():
    global running

    with st.sidebar:
        selected = option_menu(
        menu_title=None,
        options=["Analisis Data", "Emotion Cam"]
    )
        
    if selected == "Analisis Data":
        st.title(":rainbow[Emotion Detection With ANN]")
        st.subheader("Hasil Analisis:")
        option = evaluate_model()

    if selected == "Emotion Cam":
        st.title(":rainbow[Emotion Detection With ANN]")
        st.subheader("Pengecekan Emosi Real Time")
        
        if st.button("Start"):
            running = True
            detect_emotion()
        
        if st.button("Stop"):
            running = False
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
