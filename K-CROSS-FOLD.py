import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('D:\\PPDM\\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Load weights into new model
def create_model():
    model = model_from_json(loaded_model_json)
    model.load_weights("D:\\PPDM\\emotion_model.weights.h5")
    return model

print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Load test data
test_dir = 'archive/test'
if not os.path.exists(test_dir):
    print(f"Test directory {test_dir} does not exist.")
    exit()

# Ensure the test directory contains class subdirectories
class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
if not class_dirs:
    print(f"No class directories found in {test_dir}.")
    exit()

# Define KFold cross-validation with k = number of classes
k = len(class_dirs)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
confusion_matrices = []
precision_scores = []
recall_scores = []
f1_scores = []

# Collect all image file paths and their corresponding labels
file_paths = []
labels = []
for class_dir in class_dirs:
    class_files = [os.path.join(test_dir, class_dir, fname) for fname in os.listdir(os.path.join(test_dir, class_dir))]
    file_paths.extend(class_files)
    labels.extend([class_dir.capitalize()] * len(class_files))  # Capitalize to match emotion_dict keys

# Convert labels to string values
string_labels = [str(label) for label in labels]

# Perform KFold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(file_paths)):
    print(f"Training on fold {fold + 1}")

    # Create new model for each fold to avoid data leakage
    emotion_model = create_model()

    train_files = [file_paths[i] for i in train_index]
    val_files = [file_paths[i] for i in val_index]
    train_labels = [string_labels[i] for i in train_index]
    val_labels = [string_labels[i] for i in val_index]

    train_data = test_data_gen.flow_from_dataframe(
        pd.DataFrame({'filename': train_files, 'class': train_labels}),
        x_col='filename',
        y_col='class',
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64,
        shuffle=True
    )

    val_data = test_data_gen.flow_from_dataframe(
        pd.DataFrame({'filename': val_files, 'class': val_labels}),
        x_col='filename',
        y_col='class',
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64,
        shuffle=False
    )

    # Compile model before training
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model on the current fold
    emotion_model.fit(train_data, epochs=20, validation_data=val_data)

    # Evaluate the model on the validation set
    val_predictions = emotion_model.predict(val_data)
    val_predictions_labels = np.argmax(val_predictions, axis=1)
    true_labels = val_data.classes

    # Store evaluation metrics
    cm = confusion_matrix(true_labels, val_predictions_labels)
    confusion_matrices.append(cm)
    precision = precision_score(true_labels, val_predictions_labels, average='weighted')
    recall = recall_score(true_labels, val_predictions_labels, average='weighted')
    f1 = f1_score(true_labels, val_predictions_labels, average='weighted')
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Display metrics for this fold
    print(f"Confusion Matrix for fold {fold + 1}:\n", cm)
    print(f"Precision Score for fold {fold + 1}: ", precision)
    print(f"Recall Score for fold {fold + 1}: ", recall)
    print(f"F1 Score for fold {fold + 1}: ", f1)

# Calculate and display average metrics over all folds
average_confusion_matrix = np.mean(confusion_matrices, axis=0)
average_precision_score = np.mean(precision_scores)
average_recall_score = np.mean(recall_scores)
average_f1_score = np.mean(f1_scores)

print("Average Confusion Matrix:\n", average_confusion_matrix)
print("Average Precision Score: ", average_precision_score)
print("Average Recall Score: ", average_recall_score)
print("Average F1 Score: ", average_f1_score)

# Display the average confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=average_confusion_matrix, display_labels=emotion_dict.values())
cm_display.plot()
plt.show()
