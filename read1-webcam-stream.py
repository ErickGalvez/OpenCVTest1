import cv2

# Load the pre-trained Haar cascade classifier for face detection
# You need to have the file "haarcascade_frontalface_default.xml" from OpenCV's GitHub repo.
# Download it from https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab a frame.")
        break

    # Convert the frame to grayscale (Haar cascade requires grayscale input)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find the largest face
    largest_face = None
    max_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    # Draw rectangles around all detected faces
    for (x, y, w, h) in faces:
        color = (255, 0, 0)  # Default color for faces
        if (x, y, w, h) == largest_face:
            color = (0, 255, 0)  # Highlight the largest face in green
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display "Face Detected" or "No Face Detected"
    if len(faces) > 0:
        cv2.putText(frame, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()