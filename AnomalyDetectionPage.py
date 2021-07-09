import tkinter as tk

from GUI import GUI


class AnomalyDetection(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Anomaly Detection")
        label1.pack(side="top")

        label2 = tk.Label(self.main_frame, font=("Verdana", 10), text="Coming Soon...", fg='blue4')
        label2.pack(side="top")



