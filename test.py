import csv
from FaceDetect.simple_facerec import SimpleFacerec
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from Deepsort.deepsort import run_realtime_tracking
import argparse
import socket
import pickle
from dollarpy import Recognizer
from Movement_Classification.FaceExp_MovementClass import HandGestureCapture
import speech_recognition as sr

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition CRUD")

        self.create_gui()

        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("D:/HCI/SmartKitchen/FaceDetect/faces")

        self.cap = cv2.VideoCapture(0)
        self.update_video()
        self.food_prefs={}
        # Initialize the recognizer for speech recognition
        self.recognizer = sr.Recognizer()
        self.speech_button_pressed = False
        self.after_id = None







    def create_gui(self):
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(sticky="nsew")

        # Labels and Entry widgets for person information
        ttk.Label(self.frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.name_entry = ttk.Entry(self.frame)
        self.name_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(self.frame, text="Age:").grid(row=1, column=0, sticky=tk.W)
        self.age_entry = ttk.Entry(self.frame)
        self.age_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E))

        # Favorite Food 1
        ttk.Label(self.frame, text="Favorite Food 1:").grid(row=3, column=0)
        self.food1 = ttk.Entry(self.frame)
        self.food1.grid(row=3, column=1)

        # Favorite Food 2
        ttk.Label(self.frame, text="Favorite Food 2:").grid(row=4, column=0)
        self.food2 = ttk.Entry(self.frame)
        self.food2.grid(row=4, column=1)

        # Favorite Food 3
        ttk.Label(self.frame, text="Favorite Food 3:").grid(row=5, column=0)
        self.food3 = ttk.Entry(self.frame)
        self.food3.grid(row=5, column=1)

        #button to create person favourite foods
        create_button=ttk.Button(self.frame, text="create", command=self.create_person)
        create_button.grid(row=2, column=1, columnspan=3, pady=10)

        # Read button for reading favorite foods
        read_button = ttk.Button(self.frame, text="Read", command=self.read_person_info)
        read_button.grid(row=2, column=4, columnspan=3, pady=10)

        # Update button for updating favourite foods
        self.update_button = ttk.Button(self.frame, text="Update", command=self.update_person_info)
        self.update_button.grid(row=2, column=7, columnspan=3, pady=10)

        # button to delete user
        create_button = ttk.Button(self.frame, text="delete", command=self.delete_person_info)
        create_button.grid(row=2, column=10, columnspan=3, pady=10)



        # button to delete user
        create_button = ttk.Button(self.frame, text="Add Kitchen Items", command=self.ObjectDetection_tracking)
        create_button.grid(row=2, column=16, columnspan=3, pady=10)

        create_button = ttk.Button(self.frame, text="show kitchen items", command=self.show_kitchen_items)
        create_button.grid(row=2, column=19, columnspan=3, pady=10)

        create_button = ttk.Button(self.frame, text="Let's Cook", command=self.face_expression)
        create_button.grid(row=2, column=22, columnspan=3, pady=10)
        # Canvas for displaying video feed
        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=1, column=0)
        # Button to start speech recognition
        self.speech_button = ttk.Button(self.frame, text="Start Speech Recognition",
                                        command=self.toggle_speech_recognition)
        self.speech_button.grid(row=2, column=25, columnspan=3, pady=10)
        create_button = ttk.Button(self.frame, text="AR detection", command=self.connect_socket)
        create_button.grid(row=2, column=13, columnspan=3, pady=10)

    def update_person_info(self):
        name = self.name_entry.get()
        new_age = self.age_entry.get()
        new_food1 = self.food1.get()
        new_food2 = self.food2.get()
        new_food3 = self.food3.get()

        # Read the current entries from the CSV file
        rows = []
        with open('foods.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:  # make sure the row is not empty
                    rows.append(row)

            updated = False
            for row in rows:
                if row[0] == name:
                    if new_age:
                        row[1] = new_age
                    if new_food1:
                        row[2] = new_food1
                    if new_food2:
                        row[3] = new_food2
                    if new_food3:
                        row[4] = new_food3
                    updated = True
                    break
                    updated = True
                    break

            if updated:
                # Write the updated data back to the CSV file
                with open('foods.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(rows)
                print(f"Updated favorite foods for {name}")
            else:
                print(f"No entry found for {name}. Use 'Create' to add a new entry.")

                # Clear the entry fields except for the name field

        self.age_entry.delete(0, 'end')
        self.food1.delete(0, 'end')
        self.food2.delete(0, 'end')
        self.food3.delete(0, 'end')

    def show_kitchen_items(self):
        with open('D:/HCI/SmartKitchen/Deepsort/outputs/detected_classes.txt', 'r') as file:
            items = file.read().splitlines()

        # Create a new window to display the items
        root = tk.Toplevel(self.master)
        root.title("Detected Kitchen Items")

        items_label = tk.Label(root, text="Detected Kitchen Items:")
        items_label.pack()

        for item in items:
            label = tk.Label(root, text=item)
            label.pack()

        root.mainloop()

    def toggle_speech_recognition(self):
        self.speech_button_pressed = not self.speech_button_pressed
        if self.speech_button_pressed:
            self.after_id = self.master.after(1000, self.check_speech_input)
        else:
            self.master.after_cancel(self.after_id)

    def check_speech_input(self):
        if self.speech_button_pressed:
            with sr.Microphone() as source:
                print("Say something...")
                try:
                    audio_data = self.recognizer.listen(source)
                    command = self.recognizer.recognize_google(audio_data).lower()

                    # Check for specific commands and perform actions
                    if "click" in command and "button" in command:
                        self.perform_button_click()

                except sr.UnknownValueError:
                    print("Speech Recognition could not understand audio.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")

            self.after_id = self.master.after(1000, self.check_speech_input)

    def perform_button_click(self):
        # Add logic here to simulate a button click or perform the desired action
        print("Button clicked!")
    def read_person_info(self):
        name = self.name_entry.get()

        # Read the current entries from the CSV file
        with open('foods.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == name:
                    # Display the favorite foods for the person
                    self.age_entry.delete(0,tk.END)
                    self.age_entry.insert(0,row[1] if len(row)>1 else "")
                    self.food1.delete(0, tk.END)
                    self.food1.insert(0, row[2] if len(row) > 2 else "")
                    self.food2.delete(0, tk.END)
                    self.food2.insert(0, row[3] if len(row) > 3 else "")
                    self.food3.delete(0, tk.END)
                    self.food3.insert(0, row[4] if len(row) > 4 else "")
                    break
            else:
                print(f"No entry found for {name}.")

    def delete_person_info(self):
        name = self.name_entry.get()

        # Read the current entries from the CSV file and filter out the row for the detected name
        rows = []
        with open('foods.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] != name:  # Exclude the row for the detected name
                    rows.append(row)

        # Write the filtered data back to the CSV file
        with open('foods.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

        print(f"Deleted information for {name} from the CSV file")

        # Clear the entry fields except for the name field
        self.name_entry.delete(0,'end')
        self.age_entry.delete(0, 'end')
        self.food1.delete(0, 'end')
        self.food2.delete(0, 'end')
        self.food3.delete(0, 'end')

    def start_speech_recognition(self):
        with sr.Microphone() as source:
            print("Say something...")
            try:
                audio_data = self.recognizer.listen(source)
                command = self.recognizer.recognize_google(audio_data).lower()

                # Check for specific commands and perform actions
                if "click" in command and "button" in command:
                    self.perform_button_click()

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    def perform_button_click(self):
        # Add logic here to simulate a button click or perform the desired action
        print("Button clicked!")

    def update_video(self):

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (900, 900))
        face_locations, face_names = self.sfr.detect_known_faces(frame)

        if len(face_names) > 0:
            name = face_names[0]
            # Display favorite food



        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

            # Display name
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, name)



        self.canvas.config(width=900, height=900)
        self.photo = self.convert_frame_to_photo(frame)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.master.after(10, self.update_video)


    def user_exists(self, name):
        # Check if the user already exists in the CSV file
        with open('foods.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == name:
                    return True
        return False




    def create_person(self):
        # Get person information from the GUI
        name = self.name_entry.get()

        # Check if the user already exists
        if self.user_exists(name):
            self.display_message(f"The user '{name}' already exists try read it.")
        else:
            # Get the rest of the person's information
            age = self.age_entry.get()
            food1 = self.food1.get()
            food2 = self.food2.get()
            food3 = self.food3.get()

            # Save the person's information to the CSV file
            self.save_to_csv(name, age, food1, food2, food3)
            print("Added favorite foods for", name)

            # Clear the entry fields except for the name field

        self.age_entry.delete(0, 'end')
        self.food1.delete(0, 'end')
        self.food2.delete(0, 'end')
        self.food3.delete(0, 'end')





    def display_message(self, message):
        # Create a new window to display the message
        message_window = tk.Toplevel(self.master)
        message_window.title("Message")
        label = tk.Label(message_window, text=message)
        label.pack()




    def convert_frame_to_photo(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        return photo

    def connect_socket(self):

        HOST = '192.168.100.9'  # Listen on all network interfaces
        PORT = 12552  # Port number

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)  # Listen for incoming connections (1 allowed)

        print("Python Server is listening...")

        client_socket, addr = server_socket.accept()
        print("Connected by:", addr)

        # Receive data from Unity
        received_data = client_socket.recv(1024)
        print("Received from Unity:", received_data.decode())

        # send data to unity
        data_to_send = "Hello from Python Server!"
        client_socket.sendall(data_to_send.encode())

        client_socket.close()
        server_socket.close()

    def ObjectDetection_tracking(self):

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--imgsz',
            default=None,
            help='image resize, 640 will resize images to 640x640',
            type=int
        )
        parser.add_argument(
            '--model',
            default='fasterrcnn_resnet50_fpn_v2',
            help='model name',
            choices=[
                'fasterrcnn_resnet50_fpn_v2',
                'fasterrcnn_resnet50_fpn',
                'fasterrcnn_mobilenet_v3_large_fpn',
                'fasterrcnn_mobilenet_v3_large_320_fpn',
                'fcos_resnet50_fpn',
                'ssd300_vgg16',
                'ssdlite320_mobilenet_v3_large',
                'retinanet_resnet50_fpn',
                'retinanet_resnet50_fpn_v2'
            ]
        )
        parser.add_argument(
            '--threshold',
            default=0.8,
            help='score threshold to filter out detections',
            type=float
        )
        parser.add_argument(
            '--embedder',
            default='mobilenet',
            help='type of feature extractor to use',
            choices=[
                "mobilenet",
                "torchreid",
                "clip_RN50",
                "clip_RN101",
                "clip_RN50x4",
                "clip_RN50x16",
                "clip_ViT-B/32",
                "clip_ViT-B/16"
            ]
        )
        parser.add_argument(
            '--show',
            action='store_true',
            help='visualize results in real-time on screen'
        )
        parser.add_argument(
            '--cls',
            nargs='+',
            default=list(range(91)),
            help='which classes to track',
            type=int
        )
        args = parser.parse_args()

        run_realtime_tracking(args)

    def face_expression(self):

        # Load the templates array from the file
        with open('D:/HCI/SmartKitchen/templates.pkl', 'rb') as file:
            loaded_templates = pickle.load(file)
        # Create a recognizer and use it to classify gestures
        recognizer = Recognizer(loaded_templates)
        hand_gesture_capture = HandGestureCapture()
        points_say_hello = hand_gesture_capture.capture_video_points()
        result = recognizer.recognize(points_say_hello)

        print(result[0])


    def __del__(self):
        self.cap.release()

    def save_to_csv(self, name,age, food1, food2, food3):

        with open('foods.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [name,age, food1, food2, food3]
            writer.writerow(row)




if __name__ == "__main__":

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()