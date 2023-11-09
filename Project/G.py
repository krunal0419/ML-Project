import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
from PIL import Image, ImageTk

def start_capture():
    global capturing, face_data
    capturing = True
    messagebox.showinfo("Info", "Capture started")

def stop_capture():
    global capturing, face_data
    capturing = False
    file_name = entry.get()
    if len(face_data) > 0:
        face_data_np = np.asarray(face_data)
        face_data_np = face_data_np.reshape((face_data_np.shape[0], -1))
        file_path = dataset_path + file_name + '.npy'
        np.save(file_path, face_data_np)
        messagebox.showinfo("Info", "Data successfully saved at " + file_path)
        print("Data successfully saved at " + file_path)
    else:
        messagebox.showerror("Error", "No face data captured!")

def enable_start_button():
    name = entry.get()
    start_button.config(state=tk.NORMAL if name else tk.DISABLED)

def capture_frames():
    global capturing, face_data
    frame_count = 0  # Initialize frame count
    while capturing:
        ret, frame = cap.read()
        if ret == False:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) == 0:
            continue
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face_section = frame[y:y + h, x:x + w]
            face_section = cv2.resize(face_section, (100, 100))
            face_data.append(face_section)
        # Resize the frame to fit inside the tkinter canvas
        frame = cv2.resize(frame, (640, 480))
        display_frame(frame)
        frame_count += 1  # Increment frame count
        print("Frame Count:", frame_count)  # Print frame count to terminal
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break
    cap.release()

def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    canvas.create_image(0, 0, anchor=tk.NW, image=frame_photo)
    canvas.image = frame_photo

# Initialize capturing as False
capturing = False
face_data = []

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = './data/'

# Create the main window
root = tk.Tk()
root.title("Face Data Capture")
root.geometry("800x600")

# Use a custom icon
root.iconbitmap("icon.ico")

# Create and place a label
label = tk.Label(root, text="Enter the name of the person:")
label.pack(pady=10)

# Create and place an entry widget
entry = tk.Entry(root, font=("Helvetica", 16))
entry.pack(pady=10)

# Create and place start and stop buttons with custom styles
start_button = tk.Button(root, text="Start Capture", command=lambda: [start_capture(), threading.Thread(target=capture_frames).start()],
                         font=("Helvetica", 14), bg="green", fg="white")
start_button.pack(pady=10)

# Create a function to enable the "Start Capture" button when a name is entered
entry.bind("<KeyRelease>", lambda event: enable_start_button())

# Disable the "Start Capture" button by default
start_button.config(state=tk.DISABLED)

stop_button = tk.Button(root, text="Stop Capture", command=stop_capture, font=("Helvetica", 14), bg="red", fg="white")
stop_button.pack(pady=10)

# Create a canvas to display the OpenCV frames
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

root.mainloop()
