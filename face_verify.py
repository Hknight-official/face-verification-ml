import os
from sklearn.preprocessing import LabelEncoder
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern

model = joblib.load('face_verify_model.pkl')
threshold = 0.5
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys'):
    hog_features = []
    for img in images:
        features = hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                       orientations=orientations, block_norm=block_norm, visualize=False)
        hog_features.append(features)
    return np.array(hog_features)


def extract_lbp_features(images, P=8, R=1):
    lbp_features = []
    for img in images:
        # Convert image to grayscale if not already
        # if len(img.shape) == 3:
        #     img = rgb2gray(img)
        #
        # Extract LBP features
        lbp = local_binary_pattern(img, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float")
        hist /= hist.sum()  # Normalize the histogram
        lbp_features.append(hist)
    return np.array(lbp_features)


def load_name(folder):
    labels = []
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    labels.append(person_name)
    return np.array(labels)


labels = load_name('./data')
label_encoder = LabelEncoder()
labels_num = label_encoder.fit_transform(labels)

cap = cv2.VideoCapture(0)


def detect_and_predict():
    verify_status = False
    name = input("Enter your name: ")
    predicts_true = np.array([0])
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if verify_status:
            cv2.putText(frame, 'Verification Completed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_roi, (100, 100))
                hog_features = extract_lbp_features([resized_face])
                prediction = model.predict(hog_features)
                prob = model.predict_proba(hog_features)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                pre_name = label_encoder.inverse_transform(prediction)[0]
                print(pre_name, np.max(prob))
                cv2.putText(frame, str(np.sum(predicts_true == 1) / predicts_true.size), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)
                if pre_name == name and not verify_status:
                    predicts_true = np.append(predicts_true, 1)

                    if np.sum(predicts_true == 1) / predicts_true.size > 0.6 and predicts_true.size > 30:
                        verify_status = True
                else:
                    predicts_true = np.append(predicts_true, 0)
        cv2.imshow('Face Detection and Prediction', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Stop.")
            break

        if key & 0xFF == ord('r'):
            print("Reload verify result.")
            predicts_true = np.array([0])
            verify_status = False

    cap.release()
    cv2.destroyAllWindows()


detect_and_predict()
