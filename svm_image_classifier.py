import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


DATADIR = "PetImages"


CATEGORIES = ["Cat", "Dog"]

data = []
labels = []


for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    print("Loading:", path)

    for img in os.listdir(path)[:1000]:
        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))

            data.append(image.flatten())
            labels.append(label)

        except:
            pass


X = np.array(data)
y = np.array(labels)

print("Total Images Loaded:", len(X))
print("Unique Classes:", np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = SVC(kernel='linear')

model.fit(X_train, y_train)

print("Model Trained Successfully!")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 2))


plt.bar(['SVM Accuracy'], [accuracy])

plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.title("Cat vs Dog Classification Accuracy")

plt.show()