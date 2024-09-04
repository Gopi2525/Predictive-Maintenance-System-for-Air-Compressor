
import serial

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier





# Read the dataset
data = pd.read_csv(r'C:\Users\gopim\Downloads\newpulse.csv')

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create a KNN classifier object
k = 4
clf = KNeighborsClassifier(n_neighbors=k)
# Train the classifier on the data
clf.fit(X, y)




# Open the serial port
with serial.Serial('COM3', 115200) as ser:

        for i in range(50000):
            if i > 5:
                # line = ser.readline().decode('utf-8').strip()
                line = ser.readline()
                if line:

                        # Process the input data
                        input_data = np.array([int(x) for x in line.split()], dtype=int).reshape(1, -1)

                        # Make predictions
                        prediction = clf.predict(input_data)
                        print(prediction)



