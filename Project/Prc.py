import tkinter as tk

# Function to open the data generation window
def open_data_generation_window():
    data_generation_window = tk.Toplevel(root)
    data_generation_window.title("Data Generation")
    
import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person : ")
while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()


# Function to open the classification window
def open_classification_window():
    classification_window = tk.Toplevel(root)
    classification_window.title("Face Classification")
    import numpy as np
import cv2
import os
import tkinter as tk
import threading
from PIL import Image, ImageTk

# ... (KNN code and functions)
def distance(v1, v2):
	# Euclidean 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

# Global variable to control the classification process
stop_classification_event = threading.Event()
current_frame = None

# Function to start classification
def start_classification():
    def classification_thread():
        global current_frame
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        skip = 0
        dataset_path = './data/'

        face_data = [] 
        labels = []

        class_id = 0 # Labels for the given file
        names = {} # Mapping between id - name

        # Data Preparation
        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                # Create a mapping between class_id and name
                names[class_id] = fx[:-4]
                print("Loaded " + fx)
                data_item = np.load(dataset_path + fx)
                face_data.append(data_item)

                # Create Labels for the class
                target = class_id * np.ones((data_item.shape[0],))
                class_id += 1
                labels.append(target)

        face_dataset = np.concatenate(face_data, axis=0)
        face_labels = np.concatenate(labels, axis=0).reshape((-1,1))

        print(face_dataset.shape)
        print(face_labels.shape)

        trainset = np.concatenate((face_dataset, face_labels), axis=1)
        print(trainset.shape)

        while not stop_classification_event.is_set():
            ret, frame = cap.read()
            if ret == False:
                continue

            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            if len(faces) == 0:
                continue

            for face in faces:
                x, y, w, h = face

                # Get the face ROI
                offset = 10
                face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]

                # Check if the face_section is not empty before resizing
                if face_section.shape[0] > 0 and face_section.shape[1] > 0:
                    face_section = cv2.resize(face_section, (100, 100))
                else:
                    # Handle the case where the ROI is empty (e.g., no face detected)
                    continue

                # Predicted Label (out)
                out = knn(trainset, face_section.flatten())

                # Display on the screen the name and rectangle around it
                pred_name = names[int(out)]
                cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=classification_thread).start()

# Function to stop classification
def stop_classification():
    stop_classification_event.set()

# Function to update the tkinter canvas with the OpenCV frame
def update_canvas():
    if current_frame is not None:
        img = Image.fromarray(current_frame)
        img = ImageTk.PhotoImage(image=img)
        canvas.config(image=img)
        canvas.image = img
    root.after(10, update_canvas)

# Create a tkinter window
root = tk.Tk()
root.title("Face Classification")
root.geometry("800x600")

# Use a custom icon
root.iconbitmap("icon.ico")

gray_frame = tk.Frame(root, bg="gray")
gray_frame.place(relx=0, rely=0, relwidth=2, relheight=2)

# Create a button to start classification with a green background
start_button = tk.Button(root, text="Start Classification", command=start_classification, font=("Helvetica", 16), bg="green", fg="white")
start_button.pack(pady=10)

# Create a button to stop classification with a red background
stop_button = tk.Button(root, text="Stop Classification", command=stop_classification, font=("Helvetica", 16), bg="red", fg="white")
stop_button.pack(pady=10)

# Create a canvas to display OpenCV output
canvas = tk.Label(root, width=640, height=480)
canvas.pack()

# Start the function to update the canvas
update_canvas()

root.mainloop()


# Create the main tkinter window
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("400x200")

# Create labels and buttons
label = tk.Label(root, text="Choose an option:", font=("Helvetica", 16))
label.pack(pady=20)

generate_data_button = tk.Button(root, text="Generate Data", command=open_data_generation_window, font=("Helvetica", 14))
generate_data_button.pack(pady=10)

classify_button = tk.Button(root, text="Classify Faces", command=open_classification_window, font=("Helvetica", 14))
classify_button.pack(pady=10)

# Start the tkinter main loop
root.mainloop()
