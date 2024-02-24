import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect face and predict mask
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Simple rule-based method for mask detection (you can replace this with a machine learning model)
        mask_color = (0, 255, 0)  # Green color for "Mask"
        no_mask_color = (0, 0, 255)  # Red color for "No Mask"

        cv2.putText(frame, "No Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, no_mask_color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), no_mask_color, 2)

    return frame

# Function to capture video from webcam
def webcam_mask_detection():
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = detect_mask(frame)

        cv2.imshow('Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam mask detection function
webcam_mask_detection()
