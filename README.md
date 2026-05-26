# Dogs vs Cats Classification using SVM

## Objective
Implement a Support Vector Machine (SVM) model to classify images of cats and dogs from the Kaggle dataset.

---

## Dataset
The dataset used for this project is the Dogs vs Cats dataset from Kaggle.

It contains:
- Cat images
- Dog images

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Machine Learning Algorithm
- Support Vector Machine (SVM)

SVM classification function:

f(x) = sign(w · x + b)

---

## Project Workflow
1. Load image dataset
2. Resize images using OpenCV
3. Convert images into numerical arrays
4. Split dataset into training and testing sets
5. Train SVM classifier
6. Predict cat and dog images
7. Evaluate model accuracy

---

## Folder Structure

Cats-Dogs-Classification/
│
├── PetImages/
│   ├── Cat/
│   ├── Dog/
│
├── svm_image_classifier.py
├── README.md

---

## Output
The model successfully classifies cat and dog images and displays prediction accuracy.

Example Output:

Model Trained Successfully!
Accuracy: 0.53

---

## How to Run the Project

### Install Required Libraries

```bash
python -m pip install numpy scikit-learn opencv-python matplotlib