import pickle
from dollarpy import Recognizer
import pyglet

from FaceDetect.FaceDetection import FaceRecognitionApp
import tkinter as tk
from  Movement_Classification.Dynamic_Movement_Classification import CaptureVedioPoints
# Load the templates array from the file
with open('templates.pkl', 'rb') as file:
    loaded_templates = pickle.load(file)


Hello_sound = pyglet.media.load("D:/HCI/SmartKitchen/SoundEffects/hello,there.mp3", streaming=False)













# Create a recognizer and use it to classify gestures
recognizer = Recognizer(loaded_templates)

points_say_hello = CaptureVedioPoints()
result = recognizer.recognize(points_say_hello)

print(result[0])






if (result[0] == 'Hello'):
    # Load Hello sounds
    Hello_sound.play()
    # connect()

    # Create the Tkinter root window
    root = tk.Tk()
    # Create an instance of the FaceRecognitionApp class, passing the root window as the master
    app = FaceRecognitionApp(root)
    # Start the Tkinter event loop to run the GUI application
    root.mainloop()



    # hyb2a feh zorar fl gui esmo lets cook msln awl ndos 3leh b3d lma azwd kol el data bta3t kol user w el prefrences
    #bwydeny l arcore w awrelo swr akl ana 3aiz a3mlha w hoa bytl3 el data bt3tha






