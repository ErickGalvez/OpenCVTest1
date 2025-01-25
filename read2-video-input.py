import cv2

# Load the pre-trained Haar cascade classifiers for face and dog detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#dog_cascade = cv2.CascadeClassifier('dog_face_haar_cascade.xml')  # Replace with the correct path to your dog cascade file

# Allow video as input
input_source = input("Enter the video file path (or press Enter to use the webcam): ")
if input_source.strip() == "":
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(input_source)

if not video_capture.isOpened():
    print("Error: Could not access the video source.")
    exit()

# Get video properties for output
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
output_file = "output_with_detections.avi"

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab a frame or end of video reached.")
        break

    # Convert the frame to grayscale (Haar cascade requires grayscale input)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect dogs in the frame with adjusted parameters
    #dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

    # Find the largest face
    largest_face = None
    max_face_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > max_face_area:
            max_face_area = area
            largest_face = (x, y, w, h)

    # Draw rectangles around all detected faces
    for (x, y, w, h) in faces:
        color = (255, 0, 0)  # Default color for faces
        if (x, y, w, h) == largest_face:
            color = (0, 255, 0)  # Highlight the largest face in green
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display "Object Detected" or "No Object Detected"
    if len(faces) > 0:
        cv2.putText(frame, "Object Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Object Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Face and Dog Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam/video and output file, and close all OpenCV windows
video_capture.release()
out.release()
cv2.destroyAllWindows()

print(f"Video with detections saved as {output_file}.")
