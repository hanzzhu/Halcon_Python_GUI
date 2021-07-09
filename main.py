# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:53:23 2021

@author: 930415
"""
import tkinter as tk
from AnomalyDetectionPage import AnomalyDetection
from GUI import GUI
from ClassificationPage import Classification

from MenuBar import MenuBar
from ObjectDetectionPage import ObjectDetection
from OpenNewWindowPage import OpenNewWindow
import halcon as ha


class root(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        main_frame = tk.Frame(self, bg="#fedfed", height=600, width=1024)
        main_frame.pack_propagate(0)
        main_frame.pack(fill="both", expand="true")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        self.title("Halcon Interface")
        # self.resizable(0, 0) prevents the app from being resized
        # self.geometry("1024x600") fixes the applications size
        self.frames = {}
        pages = (Classification, ObjectDetection, AnomalyDetection)
        for F in pages:
            frame = F(main_frame, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Classification)
        menubar = MenuBar(self)
        tk.Tk.config(self, menu=menubar)

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()

    def OpenNewWindow(self):
        OpenNewWindow()

    def Quit_application(self):
        self.destroy()


class EmptyPage(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Page Two")
        label1.pack(side="top")


if __name__ == "__main__":
    top = root()
    root = root()
    root.withdraw()
    root.wm_geometry("600x640")
    root.minsize(100, 100)
    root.title("Halcon")

    root.mainloop()

# What's next?
# operator from halcon to make the training process stop/pause by pressing another button
