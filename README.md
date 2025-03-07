# Facial Recognition System  

## 🚀 Overview  
An intelligent **Face Recognition System** for real-time identification and verification using machine learning.  
## ✨ Includes

🔹 Image-based Face Recognition: Compare a test image with a dataset of known faces.<br>
🔹 Live Camera Face Recognition: Detect and recognize faces in real-time.<br>
🔹 Face Lock System: Grant access based on recognized faces.<br>
🔹 Registering New(unknown) Face Dynamically.


## 🛠️ Tech Stack  
🔹 **Languages**: Python <br>
🔹 **Libraries**: OpenCV, face_recognition, NumPy, Matplotlib,  

## 📦 Installation  
Clone the repo: 
```sh
git clone https://github.com/Siddhesh-04/CodeClause-Internship-Face-Recognition-System.git
```
```
cd face-recognition-system
```
## install dependencies: 
```bash
pip install opencv-python numpy face-recognition matplotlib pillow  
```
## 🎯 Usage
### 1. Load Known Faces
Store known faces inside the sample_images/ directory (JPG, PNG, or JPEG format). Each filename should be the name of the person.

### 2. Run the Face Recognition System
 Run the script to recognize faces from images, live camera, or the face lock system.<br>
 You can also run it in Jupyter Notebook with a .ipynb extension using a Boolean variable like IN_JUPYTER = True or False.
```bash
python face_rec.py
```
### 3. Functionality
Enter Choice to Run/call function
```
[1] Recognize Face in Image - [sample_faces/user1.jpg (Your Image Path)]
[2] Live Face Recognition - Real-time face recognition 🎥
[3] Register New Face - Dynamically Adding new user/face to dataset
[4] Liveness Detection
[5] FACE LOCK SIMULATION 
[6] Quit/Exit
Enter your choice: 
```
## Directory Structure
```
Face-Recognition-System/
│── sample_face/         # Test images for face recognition with Image
│── sample_images/       # Folder containing known face images
│── face_recognition.py  # Main script
│── README.md            # Documentation
```
🔮 Future Enhancements
```
🚀 Improve recognition accuracy using deep learning
🌐 Web/App GUI integration
🔐 Smart lock hardware integration
```
## License
This project is open-source and available for modification and improvement.

---
💡 **Tip:** Press 'q' to quit the live face recognition or face lock system.
