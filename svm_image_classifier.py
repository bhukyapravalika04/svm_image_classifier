import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

images = []
labels = []

categories = ['Cat', 'Dog']

# Load dataset
for category in categories:

    folder_path = os.path.join('PetImages', category)

    label = categories.index(category)

    print("Loading:", folder_path)

    for filename in os.listdir(folder_path)[:1000]:

        path = os.path.join(folder_path, filename)

        try:
            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            img = img.flatten()

            images.append(img)
            labels.append(label)

        except:
            pass

# Convert arrays
X = np.array(images)
y = np.array(labels)

print("Total Images Loaded:", len(X))
print("Unique Classes:", np.unique(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='linear')

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Trained Successfully!")
print("Accuracy:", accuracy)