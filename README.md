# Face Recognition System

This project implements a face recognition system using OpenCV and the `face_recognition` library. It includes features such as:
- **Image-based Face Recognition**: Compare a test image with a dataset of known faces.
- **Live Camera Face Recognition**: Detect and recognize faces in real-time.
- **Face Lock System**: Grant access based on recognized faces.

## Features
✅ **Load Known Faces**: Automatically loads faces from a folder and encodes them.<br>
✅ **Recognize Faces in Images**: Compare a test image with stored face encodings.<br>
✅ **Live Face Recognition**: Detect faces in real-time via webcam.<br>
✅ **Face Lock System**: Grants access if a recognized face is detected.

## Installation

### Prerequisites
Make sure you have Python installed. Then, install the required dependencies:
```bash
pip install opencv-python numpy face-recognition matplotlib pillow
```

## Usage

### 1. Load Known Faces
Store known faces inside the `sample_images/` directory (JPG, PNG, or JPEG format). Each filename should be the name of the person.

### 2. Run the Face Recognition System
Run the script to recognize faces from images, live camera, or the face lock system.
```bash
python face_recognition.py
```

### 3. Functionality

#### Recognize a Face from an Image
Uncomment the following line in `face_recognition.py` and run the script:
```python
recognize_faces_from_image(known_face_encodings, known_face_names, "sample_images/test.jpg")
```

#### Live Face Recognition
Run face recognition in a webcam feed:
```python
live_face_recognition(known_face_encodings, known_face_names, in_jupyter=False)
```
Set `in_jupyter=True` if running inside a Jupyter Notebook.

#### Face Lock System
Run a face lock system simulation:
```python
face_lock_system(known_face_encodings, known_face_names, in_jupyter=False)
```

## Directory Structure
```
Face-Recognition-System/
│── sample_images/       # Folder containing known face images
│── face_recognition.py  # Main script
│── README.md            # Documentation
```

## Future Improvements
🚀 Improve recognition accuracy using deep learning
🚀 Implement multi-face tracking in real-time
🚀 Enhance security measures for the face lock system

## License
This project is open-source and available for modification and improvement.

---
💡 **Tip:** Press 'q' to quit the live face recognition or face lock system.
