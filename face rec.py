import cv2
import numpy as np
import os

# Define paths
dataset_path = "dataset"
recognizer_model_path = "face_recognizer.yml"

# Create dataset folder if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize face detector and face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to capture and save images for training
def capture_images(person_name):
    cap = cv2.VideoCapture(0)
    count = 0

    person_path = os.path.join(dataset_path, person_name)
    os.makedirs(person_path, exist_ok=True)

    print(f"Capturing images for {person_name}. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            img_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            # Display the captured image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Capturing Images", frame)

        # Stop capturing after 50 images or on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {person_name}.")

# Function to train the recognizer
def train_recognizer():
    faces = []
    labels = []
    label_map = {}

    # Assign a numeric label to each person
    label_id = 0
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_map[label_id] = person_name
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(np.array(gray_image, dtype=np.uint8))
            labels.append(label_id)

        label_id += 1

    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    recognizer.save(recognizer_model_path)
    print("Training complete and model saved.")

    return label_map

# Function to recognize faces in real-time
def recognize_faces(label_map):
    cap = cv2.VideoCapture(0)

    print("Starting face recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)

            if confidence < 100:
                name = label_map[label]
                confidence_text = f"{int(100 - confidence)}%"
            else:
                name = "UNKNOWN"
                confidence_text = ""

            # Display the results
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to orchestrate capturing, training, and recognizing
if __name__ == "__main__":
    # Capture images for training (Add names here to capture for multiple people)
    person_names = ["sabbu"]
    for person_name in person_names:
        capture_images(person_name)

    # Train recognizer
    label_map = train_recognizer()

    # Recognize faces
    recognize_faces(label_map)