import tkinter as tk

from AnomalyDetectionPage import AnomalyDetection
from ClassificationPage import Classification

from ObjectDetectionPage import ObjectDetection


class MenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)

        menu_file = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Options", menu=menu_file)
        menu_file.add_command(label="Classification", command=lambda: parent.show_frame(Classification))
        menu_file.add_separator()
        menu_file.add_command(label="Object Detection", command=lambda: parent.show_frame(ObjectDetection))
        menu_file.add_separator()
        menu_file.add_command(label="Anomaly Detection", command=lambda: parent.show_frame(AnomalyDetection))
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=lambda: parent.Quit_application())

        menu_help = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Help", menu=menu_help)
        menu_help.add_command(label="Help", command=lambda: parent.OpenNewWindow())
        menu_help.add_command(label="About", command=lambda: parent.OpenNewWindow())

        ImWidth_var = tk.IntVar()
        ImHeight_var = tk.IntVar()
        ImChannel_var = tk.IntVar()
        BatchSize_var = tk.StringVar()
        InitialLearningRate_var = tk.DoubleVar()
        Momentum_var = tk.DoubleVar()
        NumEpochs_var = tk.IntVar()
        ChangeLearningRateEpochs_var = tk.StringVar()  ##
        lr_change_var = tk.StringVar()  ##
        WeightPrior_var = tk.DoubleVar()
        class_penalty_var = tk.StringVar()  ##