#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import tools as T


validation_data = np.load("../concat100/validation_data.npy")  # Load validation data from .npy file
validation_label = np.load("../concat100/validation_label.npy")  # Load validation labels from .npy file

# normalization
validation_data = validation_data / 255.0


# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = T.separate_data(validation_data, validation_label)  # Separate validation data and labels into classes

# number of class
num_classes = 5  # Cloudy,Sunny,Rainy,Snowy,Foggy

# for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)  # Convert labels to categorical format


# loads trained model and architecture
model = load_model("modelsCNN/size100/trainedModelE20.h5")  # Load the trained CNN model


# -------predicting part-------
y = model.predict_classes(validation_data, verbose=0)  # Predict classes for validation data
acc = T.get_accuracy_of_class(T.binary_to_class(validation_label), y)  # Calculate accuracy of predictions
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):  # Iterate through each class for accuracy calculation
    v_data = vd[i][0]  # Get validation data for the current class
    v_label = vd[i][1]
    y = model.predict_classes(v_data, verbose=0)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + T.classes[i] + ": ", acc)
    print("-----------------------------")