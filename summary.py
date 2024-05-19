import numpy as np
from keras.models import model_from_json

json_file = open('D:\PPDM\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:\PPDM\emotion_model.weights.h5")
print("Loaded model from disk")

emotion_model.summary()