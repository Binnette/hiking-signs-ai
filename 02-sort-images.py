# apt install python3-tk python3-pil python3-pil.imagetk
import os
import shutil
from tkinter import Tk, Label, Text, Button, Frame
from PIL import Image, ImageTk

# Function to handle the 'Yes' button click or left arrow key press
def yes_action(event=None):  # Add 'event=None' to handle the key press event
    image_path = image_list[image_index]
    try:
        shutil.move(image_path, 'signToCheck')  # Move the image
    except Exception as e:
        print(f"Error on file: {image_path} => {e}")
    root.after(0, update_image)  # Schedule the next update

# Function to handle the 'No' button click or right arrow key press
def no_action(event=None):  # Add 'event=None' to handle the key press event
    global image_index
    image_path = image_list[image_index]
    try:
        shutil.copy(image_path, '.')
    except Exception as e:
        print(f"Error on file: {image_path} => {e}")
    root.after(0, update_image)  # Schedule the next update

# Function to handle the 'Open' button click
def open_action():
    image_path = image_list[image_index]
    try:
        os.system(f"xdg-open {image_path}")  # Open the image with xdg-open command
    except Exception as e:
        print(f"Error opening file: {image_path} => {e}")

# Function to update the image displayed
def update_image():
    global img_size, image_index, image_label, status_label, image_name_label, root, image_list
    image_index += 1
    if image_index < len(image_list):
        image_path = image_list[image_index]
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_name_label.config(text=image_path)

            status_label.config(text=f'Image {image_index + 1}/{len(image_list)}')

            exif_orientation = image._getexif().get(0x112, 1)
            if exif_orientation == 3:
                image = image.rotate(180, expand=True)
            elif exif_orientation == 6:
                image = image.rotate(-90, expand=True)
            elif exif_orientation == 8:
                image = image.rotate(90, expand=True)

            image.thumbnail(img_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            # update root and center root in screen
            root.update()
            x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
            y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
            root.geometry("+%d+%d" % (x, y))


        else:
            root.after(0, update_image)  # Schedule the next update
    else:
        root.destroy()  # Close the application if there are no more images

# Read the list of images from a text file
with open('images.txt', 'r') as file:
    image_list = [line.strip() for line in file.readlines()]

image_index = -1

# Set up the GUI
root = Tk()
root.title('Image Viewer')

# Top frame
top_frame = Frame(root)
top_frame.grid(row=0, column=0)


image_name_label = Label(top_frame)
image_name_label.grid(row=0, column=0)

# Create a button with an Open icon
open_button = Button(top_frame, text='Open', command=open_action)
open_button.grid(row=0, column=1)

# Middle frame
middle_frame = Frame(root)
middle_frame.grid(row=1, column=0)

image_label = Label(middle_frame)
image_label.pack(expand=True)

# Bottom frame
bottom_frame = Frame(root)
bottom_frame.grid(row=2, column=0)

# Add a label with the question
question_label = Label(bottom_frame, text='Is it a hiking sign?')
question_label.grid(row=0, column=0)

status_label = Label(bottom_frame, text='')
status_label.grid(row=0, column=1)

yes_button = Button(bottom_frame, text='Yes', command=yes_action)
yes_button.grid(row=1, column=0)

no_button = Button(bottom_frame, text='No', command=no_action)
no_button.grid(row=1, column=1)

# Bind the left and right arrow keys to the 'yes_action' and 'no_action' functions
root.bind('<Left>', yes_action)  # Bind the left arrow key
root.bind('<Right>', no_action)  # Bind the right arrow key

root.update()
width = root.winfo_screenwidth()*0.85
height = root.winfo_screenheight()*0.85
img_size = (width, height)

# Initialize the first image update
update_image()

# Start the Tkinter event loop
root.mainloop()