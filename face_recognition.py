import cv2
import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output
import time

# Load known faces
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

# Find correct file format for a given name
def find_image_file(directory, name):
    for file in os.listdir(directory):
        if file.lower().startswith(name.lower()) and file.lower().endswith((".jpg", ".png", ".jpeg")):
            return os.path.join(directory, file)
    return None

#**********************************************************************************************************************************
#                                     *** Basic Image Face recognition  with dataset ***                                          |
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

#**********************************************************************************************************************************
#                                          📹 FACE RECOGNITION -from a live camera feed                                           |
#**********************************************************************************************************************************
def live_face_recognition(known_face_encodings, known_face_names, camera_index=0, in_jupyter=False):
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


#**********************************************************************************************************************************
#                                      🔒 Face Lock System                                                                        |
#**********************************************************************************************************************************
def face_lock_system(known_face_encodings, known_face_names, camera_index=0, in_jupyter=False):
    video_capture = cv2.VideoCapture(camera_index)

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

        if access_granted:
            message = f"✅ Access Granted: {detected_name}"
        else:
            message = "❌ Access Denied!"

        if in_jupyter:
            clear_output(wait=True)  # Clear previous output
            # Convert to PIL Image for Jupyter display
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display(img_pil, message)  # Show updated frame
        else:
            # Show in normal OpenCV window
            cv2.imshow('Live Face Recognition', frame)
            print(message)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
            
    
    
#**********************************************************************************************************************************
#                                              RUN THE SYSTEM -by Calling Function                                                   |
#**********************************************************************************************************************************

if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces()

# Note : Uncomment to run Image-to-Image recognition
    # recognize_faces_from_image(known_face_encodings, known_face_names, "sample_images/Leonardo DiCaprio.jpg")
    
# Note : Uncomment to run Default Face recognition with live cam Feed (TRUE/FALSE for running In-Jupyter OR External Window )
    
    # live_face_recognition(known_face_encodings, known_face_names, in_jupyter=False)
    # live_face_recognition(known_face_encodings, known_face_names, in_jupyter=True)

# Note : Uncomment to run FACE LOCK SIMULATION with Authorised Users in dataset (TRUE/FALSE for running In-Jupyter OR External Window ) 
    # face_lock_system(known_face_encodings, known_face_names, in_jupyter=False)
    face_lock_system(known_face_encodings, known_face_names, in_jupyter=True)
