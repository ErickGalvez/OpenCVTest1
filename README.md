## Open CV and Python testing

  This is a learning project, it uses opencv though its python library and access a local camera, it then proceeds to detect faces, it can also be given an already recorded .mp4 file as input, and can output a video file with the faces detection for documenting its funcionality

### How to use:

Make sure the following packages are installed in your project:


![opencv](https://github.com/user-attachments/assets/0f42fc4a-1b9b-40d1-acc4-7c116af3b0e2)

In case you have multiple cams set on your computer yo can tweak this value until you get the camera you are looking for:

    # Start the webcam
    video_capture = cv2.VideoCapture(0)
