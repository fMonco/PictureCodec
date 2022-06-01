from tkinter import *
from tkinter import messagebox
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys

def imageout():
    img = Image.open('1.png')
    img_arr = np.asarray(img)
    plt.axis('off')
    plt.imshow(img_arr)
    plt.show()

def makesomenoise():
    exec(open("noise.py").read())

def restore():
    exec(open("restore.py").read())

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root = Tk() 
root.eval('tk::PlaceWindow . center')
root.title("Picture Codec")
root.configure(bg="black")
root.geometry('150x94')


bttn = Button(root, text="Original", width = 25, borderwidth=5, font= 'Consolas 10',fg="Grey", bg="black", command = imageout)
bttn.pack()

bttn1 = Button(root, text="Noisy", width = 25, borderwidth=5, font= 'Consolas 10', fg="Grey", bg="black", command = makesomenoise)
bttn1.pack()

bttn2 = Button(root, text="Restored", width = 25, borderwidth=5, font= 'Consolas 10', fg="Grey", bg="black", command = restore)
bttn2.pack()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
sys.exit()