import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name_user = input("Enter your name:")
save_dir = '/home/hknight/PycharmProjects/cpv_project/assignment/data/' + name_user
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                face_img = gray[y:y+h, x:x+w]
                # Create a unique filename based on the current timestamp
                timestamp = int(time.time())
                face_filename = os.path.join(save_dir, f'face_{timestamp}_{i}.jpg')
                # Save the face image
                cv2.imwrite(face_filename, face_img)
                print(f'Saved {face_filename}')

    # Draw rectangles after saving the face images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
