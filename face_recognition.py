import cv2
import face_recognition
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output

# Constants
CAMERA_INDEX = 0
QUIT_KEY = ord('q')
FACE_ENCODING_FILE = 'face_encodings.pkl'

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    image_folder = "sample_images"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for file in os.listdir(image_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])  # Save filename without extension
    return known_face_encodings, known_face_names


#**********************************************************************************************************************************
#                              *** Basic Face recognition from Image & authenticating with knwon Faces ***                        |
#**********************************************************************************************************************************
def recognize_faces_from_image(known_face_encodings, known_face_names, test_image_path):
    if not os.path.exists(test_image_path):
        print("Test image not found!")
        return

    # Load the test image
    test_image = face_recognition.load_image_file(test_image_path)
    test_rgb_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Detect faces and encode
    test_encodings = face_recognition.face_encodings(test_image)
    if not test_encodings:
        print("No face detected in test image!")
        return

    test_encoding = test_encodings[0]  # Take the first face

    # Compare with known faces
    matches = face_recognition.compare_faces(known_face_encodings, test_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, test_encoding)

    if True in matches:
        best_match_index = np.argmin(face_distances)
        matched_name = known_face_names[best_match_index]

        # Find correct image file
        matched_image_path = find_image_file("sample_images", matched_name)

        if matched_image_path:
            matched_image = cv2.imread(matched_image_path)
            matched_image = cv2.resize(matched_image, (test_rgb_image.shape[1], test_rgb_image.shape[0]))
            # Convert test image to PIL format
            test_pil = Image.fromarray(test_rgb_image)
            matched_pil = Image.fromarray(matched_image)

            # Display images side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(test_pil)
            axes[0].set_title("Test Image")
            axes[0].axis("off")

            axes[1].imshow(matched_pil)
            axes[1].set_title(f"Matched: {matched_name}")
            axes[1].axis("off")

            plt.show()
        else:
            print(f"Matched with {matched_name}, but image file not found.")
    else:
        print("No match found in dataset!")

def find_image_file(directory, name):
    for file in os.listdir(directory):
        if file.lower().startswith(name.lower()) and file.lower().endswith((".jpg", ".png", ".jpeg")):
            return os.path.join(directory, file)
    return None

#**********************************************************************************************************************************
#                                           FACE RECOGNITION -from a live camera feed                                             |
#**********************************************************************************************************************************
def live_face_recognition(known_face_encodings, known_face_names, camera_index=CAMERA_INDEX, in_jupyter=False):
    video_capture = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    name = known_face_names[best_match_index]

            # Scale back face location
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top - 25), (right, top), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if in_jupyter:
            # Convert to PIL Image for Jupyter display
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            clear_output(wait=True)  # Clear previous output
            display(img_pil)  # Show updated frame
        else:
            # Show in normal OpenCV window
            cv2.imshow('Live Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
#**********************************************************************************************************************************
#                                         Face Lock Simulation                                                                    |
#**********************************************************************************************************************************

def face_lock_system(known_face_encodings, known_face_names, camera_index=CAMERA_INDEX, in_jupyter=False):
   
    video_capture = cv2.VideoCapture(camera_index)
    previous_access_granted = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        access_granted = False
        detected_name = "Unknown"

        if face_locations and face_encodings:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.5:  # Accept matches below threshold
                        detected_name = known_face_names[best_match_index]
                        access_granted = True

                # Scale face location back
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Bounding box

        if access_granted != previous_access_granted:
            if access_granted:
                message = f"✅ Access Granted: {detected_name}"
            else:
                message = "❌ Access Denied!"
            if in_jupyter:
                clear_output(wait=True)  # Clear previous output
                # Convert to PIL Image for Jupyter display
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                display(img_pil)  # Show updated frame
                print(message)
            else:
                # Show in normal OpenCV window
                print(message)
                cv2.imshow('Live Face Recognition', frame)
            previous_access_granted = access_granted

        if in_jupyter:
            clear_output(wait=True)  # Clear previous output
            # Convert to PIL Image for Jupyter display
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display(img_pil)  # Show updated frame
        else:
            # Show in normal OpenCV window
            cv2.imshow('Live Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
#**********************************************************************************************************************************
#                                      Liveliness Detection                                                                       |
#**********************************************************************************************************************************
def liveness_detection(frame):
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a Fourier Transform to the blurred frame
    fft = np.fft.fft2(blurred)

    # Shift the FFT to the center of the image
    fft_shift = np.fft.fftshift(fft)

    # Calculate the magnitude of the FFT
    magnitude = np.abs(fft_shift)

    # Calculate the phase of the FFT
    phase = np.angle(fft_shift)

    # Calculate the power spectral density of the FFT
    psd = np.abs(magnitude) ** 2

    # Calculate the average power spectral density
    avg_psd = np.mean(psd)

    # If the average power spectral density is above a certain threshold, consider the face as live
    if avg_psd > 1000:
        return True
    else:
        return False
    
#**********************************************************************************************************************************
#                                              REGISTRATION OF NEW USER/FACE                                                      |
#**********************************************************************************************************************************
def register_new_user(known_face_encodings, known_face_names, camera_index=CAMERA_INDEX, in_jupyter=False):
    video_capture = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            face_encoding = face_encodings[0]
            name = input("Enter your name: ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

            with open(FACE_ENCODING_FILE, 'wb') as f:
                pickle.dump((known_face_encodings, known_face_names), f)

            print("User  registered successfully!")
            break

        if in_jupyter:
            # Convert to PIL Image for Jupyter display
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            clear_output(wait=True)  # Clear previous output
            display(img_pil)  # Show updated frame
        else:
            # Show in normal OpenCV window
            cv2.imshow('Live Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
#**********************************************************************************************************************************
#                                              RUN THE SYSTEM -by selecting following Choices                                     |
#**********************************************************************************************************************************
if __name__ == "__main__":
    """
    Main function to call all features.
    
    Arguments:
        known_face_encodings (list): List of known face encodings.
        known_face_names (list): List of known face names.
        camera_index (int, optional): Index of the camera to use. Defaults to CAMERA_INDEX.
        in_jupyter (bool, optional): Whether to display in Jupyter notebook. Defaults to False.
    """
    known_face_encodings, known_face_names = load_known_faces()
    while True:
        print("\n[1] Recognize Face in Image")
        print("[2] Live Face Recognition")
        print("[3] Register New Face")
        print("[4] Liveness Detection")
        print("[5] FACE LOCK SIMULATION")
        print("[6] Quit/Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            image_path = input("Enter image path: ")
            recognize_faces_from_image(known_face_encodings, known_face_names, image_path)
        elif choice == "2":
            live_face_recognition(known_face_encodings, known_face_names, in_jupyter=False)
        elif choice == "3":
            register_new_user(known_face_encodings, known_face_names, in_jupyter=False)
        elif choice == "4":
            video_capture = cv2.VideoCapture(0)
            ret, frame = video_capture.read()
            video_capture.release()
            cv2.destroyAllWindows()
            liveness = liveness_detection(frame)

            if liveness ==False :
                print("Face is not live")
            else:
                print("Face is live")
        elif choice == "5":
            face_lock_system(known_face_encodings, known_face_names, in_jupyter=False)
        elif choice == "6":
            print("[INFO] Exiting...")
            break
        else:
            print("[ERROR] Invalid choice. Try again.")
