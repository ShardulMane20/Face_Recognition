import cv2
import dlib
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import font as tkFont

# Load the dlib models for face recognition
face_rec_model_path = r"D:\Projects\Python Object Detection\Shape_Pridictor\shape_predictor_68_face_landmarks.dat"
recognition_model_path = r"D:\Projects\Python Object Detection\Shape_Pridictor\dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(face_rec_model_path)
face_rec_model = dlib.face_recognition_model_v1(recognition_model_path)

def get_face_encoding(face_img, face_rect):
    shape = shape_predictor(face_img, face_rect)
    return np.array(face_rec_model.compute_face_descriptor(face_img, shape))

def load_reference_images(reference_folder):
    encodings = {}
    for person_name in os.listdir(reference_folder):
        person_folder = os.path.join(reference_folder, person_name)
        if os.path.isdir(person_folder):
            person_encodings = []
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(person_folder, filename)
                    img = cv2.imread(img_path)
                    faces = detector(img, 1)
                    if len(faces) == 1:
                        person_encodings.append(get_face_encoding(img, faces[0]))
            if person_encodings:
                encodings[person_name] = np.mean(person_encodings, axis=0)
    return encodings

def start_attendance_system():
    reference_folder = r"D:\Projects\Python Object Detection\photos"
    encodings = load_reference_images(reference_folder)

    video_capture = cv2.VideoCapture(0)

    attendance_file = r"D:\Projects\Python Object Detection\attendance.csv"
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Name, Timestamp\n")

    attended = set()
    recognized_faces = {}

    name_buffer = {}
    buffer_frames = 10
    confidence_threshold = 0.6

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        detected_faces = detector(frame, 1)

        current_faces = {}

        for face in detected_faces:
            face_encoding = get_face_encoding(frame, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            matched_name = None
            for name, ref_encoding in encodings.items():
                distance = np.linalg.norm(ref_encoding - face_encoding)
                if distance < confidence_threshold:
                    if name not in name_buffer:
                        name_buffer[name] = 1
                    else:
                        name_buffer[name] += 1
                    if name_buffer[name] >= buffer_frames:
                        matched_name = name
                        break
                else:
                    name_buffer[name] = 0

            if matched_name:
                if matched_name not in attended:
                    attended.add(matched_name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(attendance_file, 'a') as f:
                        f.write(f"{matched_name}, {timestamp}\n")
                current_faces[matched_name] = (x, y)
            else:
                matched_name = "Not recognized"

        recognized_faces = current_faces

        for name, (x, y) in recognized_faces.items():
            label = f"{name} (Attendance marked!)" if name in attended else name
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Attendance System", cv2.WND_PROP_VISIBLE) < 1:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Create the Tkinter window with an appealing design
def create_start_window():
    start_window = tk.Tk()
    start_window.title("Attendance Marking System")
    start_window.geometry("600x400")
    start_window.configure(bg="#0F1B2A")

    # Create a custom font
    custom_font_title = tkFont.Font(family="Helvetica", size=24, weight="bold")

    # Frame for content
    frame = tk.Frame(start_window, bg="#0F1B2A", padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)

    # Title label
    title_label = tk.Label(frame, text="Attendance Marking System", font=custom_font_title, bg="#0F1B2A", fg="#ECF0F1")
    title_label.pack(pady=30)

    # Start button
    start_button = tk.Button(frame, text="Start Attendance", command=lambda: [start_window.destroy(), start_attendance_system()],
                             borderwidth=0, bg="#2ECC71", fg="#FFFFFF", font=("Helvetica", 16),
                             activebackground="#27AE60", activeforeground="#FFFFFF", relief=tk.RAISED)
    start_button.pack(pady=10, padx=10)

    # Exit button
    exit_button = tk.Button(frame, text="Exit", command=start_window.quit,
                            borderwidth=0, bg="#E74C3C", fg="#FFFFFF", font=("Helvetica", 16),
                            activebackground="#C0392B", activeforeground="#FFFFFF", relief=tk.RAISED)
    exit_button.pack(pady=10, padx=10)

    # Run the Tkinter main loop
    start_window.mainloop()

# Run the start window
create_start_window()
