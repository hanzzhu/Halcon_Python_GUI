from SettingVariables import SettingVariables
import numpy
from GUI import GUI
from makeCM import make_confusion_matrix, divide_chunks, figure
import tkinter as tk
from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import threading
import os
from mttkinter import *
import halcon as ha

from makeCM import make_confusion_matrix

"""

Initialise Halcon

"""

Chadle_ProjectsDir = 'C:/Chadle_Projects'

Chadle_DataDir = Chadle_ProjectsDir + '/Chadle_Data'
Chadle_Halcon_ScriptsDir_CL = Chadle_ProjectsDir + '/Chadle_Halcon_Scripts/CL'
Halcon_DL_library_filesDir = Chadle_ProjectsDir + '/Halcon_DL_library_files'

engine = ha.HDevEngine()
engine.set_procedure_path('C:/MVTec/Halcon-20.11-Progress/procedures')
engine.set_procedure_path(Halcon_DL_library_filesDir)
# path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located

program = ha.HDevProgram(Chadle_Halcon_ScriptsDir_CL+'/DL_train_CL_seagate.hdev')
aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE_old'))
evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'Evaluation'))


class Classification(GUI):  # inherits from the GUI class

    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        styles = SettingVariables.frame_styles
        dir_color = SettingVariables.dirColor
        backgroundColor = SettingVariables.backgroundColor
        """

        Frames

        """
        frame_Parameters = tk.LabelFrame(self, styles, text="Parameters")
        frame_Parameters.place(rely=0.07, relx=0.02, height=300, width=600)

        frame_Augmentation = tk.LabelFrame(self, styles, text="Augmentation")
        frame_Augmentation.place(rely=0.47, relx=0.02, height=300, width=600)

        frameSettings = tk.LabelFrame(self, styles, text="Settings")
        frameSettings.place(rely=0.07, relx=0.45, height=180, width=700)

        frame_Inspection_graphical = tk.LabelFrame(self, styles, text="Graphical Inspection")
        frame_Inspection_graphical.place(rely=0.3, relx=0.45, height=500, width=550)

        frame_Inspection_stats = tk.LabelFrame(self, styles, text="Statistics")
        frame_Inspection_stats.place(rely=0.3, relx=0.83, height=500, width=150)

        # frameTop = tk.LabelFrame(self, styles, )
        # frameTop.place(rely=0.01, relx=0.02, height=30, width=900)

        frameBot = tk.LabelFrame(self, styles, )
        frameBot.place(rely=0.97, relx=0, height=30, width=1440)

        """

        Classification Label(Top Left)

        """

        Classification_label = tk.Label(self, text='Classification', font=('Verdana', 20, 'bold'),
                                        bg=backgroundColor)
        Classification_label.place(rely=0, relx=0.1, height=50)

        """

        Main Operation Buttons

        """

        quitEverything = ttk.Button(self, text="Quit", command=lambda: sys.exit())
        quitEverything.place(rely=0.01, relx=0.8, height=50, width=100)

        startTrainingButton = ttk.Button(self, text="Start Training  ", command=lambda: training_Run())
        startTrainingButton.place(rely=0.01, relx=0.6, height=50, width=100)

        startPreprocessButton = ttk.Button(self, text="Start Pre-Processing  ", command=lambda: preprocess_Run())
        startPreprocessButton.place(rely=0.01, relx=0.5, height=50, width=100)

        startEvaluationButton = ttk.Button(self, text="Evaluation  ", command=lambda: evalutation_Run())
        startEvaluationButton.place(rely=0.01, relx=0.7, height=50, width=100)

        """

        Inputs of Parameters
        
        Mandatory
        
        """

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

        ImWidth_label = tk.Label(frame_Parameters, text='Image Width:', font=('calibre', 10, 'bold'),
                                 bg=backgroundColor)
        ImWidth_entry = tk.Entry(frame_Parameters, textvariable=ImWidth_var, font=('calibre', 10, 'normal'))

        ImHeight_label = tk.Label(frame_Parameters, text='Image Height:', font=('calibre', 10, 'bold'),
                                  bg=backgroundColor)
        ImHeight_entry = tk.Entry(frame_Parameters, textvariable=ImHeight_var, font=('calibre', 10, 'normal'))

        ImChannel_label = tk.Label(frame_Parameters, text='Image Channel:', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        ImChannel_entry = tk.Entry(frame_Parameters, textvariable=ImChannel_var, font=('calibre', 10, 'normal'))

        BatchSize_label = tk.Label(frame_Parameters, text='Batch Size:', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        BatchSize_entry = tk.Entry(frame_Parameters, textvariable=BatchSize_var, font=('calibre', 10, 'normal'))

        InitialLearningRate_label = tk.Label(frame_Parameters, text='Initial Learning Rate:',
                                             font=('calibre', 10, 'bold'),
                                             bg=backgroundColor)
        InitialLearningRate_entry = tk.Entry(frame_Parameters, textvariable=InitialLearningRate_var,
                                             font=('calibre', 10, 'normal'))

        Momentum_label = tk.Label(frame_Parameters, text='Momentum:', font=('calibre', 10, 'bold'), bg=backgroundColor)
        Momentum_entry = tk.Entry(frame_Parameters, textvariable=Momentum_var, font=('calibre', 10, 'normal'))

        NumEpochs_label = tk.Label(frame_Parameters, text='Number of Epochs:', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        NumEpochs_entry = tk.Entry(frame_Parameters, textvariable=NumEpochs_var, font=('calibre', 10, 'normal'))

        ChangeLearningRateEpochs_label = tk.Label(frame_Parameters, text='Change Learning Rate Epochs:',
                                                  font=('calibre', 10, 'bold'), bg=backgroundColor)
        ChangeLearningRateEpochs_entry = tk.Entry(frame_Parameters, textvariable=ChangeLearningRateEpochs_var,
                                                  font=('calibre', 10, 'normal'))
        ChangeLearningRateEpochs_format = tk.Label(frame_Parameters, text='E.g. : 1, 2, 3...',
                                                   font=('calibre', 10, 'bold'), bg=backgroundColor, fg='dark red')

        lr_change_label = tk.Label(frame_Parameters, text='Learning Rate Schedule:', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        lr_change_entry = tk.Entry(frame_Parameters, textvariable=lr_change_var, font=('calibre', 10, 'normal'))
        lr_change_format = tk.Label(frame_Parameters, text='E.g. : 0.1, 0.01, 0.001', font=('calibre', 10, 'bold'),
                                    bg=backgroundColor, fg='dark red')

        WeightPrior_label = tk.Label(frame_Parameters, text='Regularisation Constant:', font=('calibre', 10, 'bold'),
                                     bg=backgroundColor)
        WeightPrior_entry = tk.Entry(frame_Parameters, textvariable=WeightPrior_var, font=('calibre', 10, 'normal'))
        WeightPrior_format = tk.Label(frame_Parameters, text='To reduce over-fitting', font=('calibre', 10, 'bold'),
                                      bg=backgroundColor, fg='dark red')

        class_penalty_label = tk.Label(frame_Parameters, text='Class Penalty:', font=('calibre', 10, 'bold'),
                                       bg=backgroundColor)
        class_penalty_entry = tk.Entry(frame_Parameters, textvariable=class_penalty_var, font=('calibre', 10, 'normal'))
        class_penalty_format = tk.Label(frame_Parameters, text='E.g. : 1, 1', font=('calibre', 10, 'bold'),
                                        bg=backgroundColor, fg='dark red')

        settingEntryLength = 6
        ImWidth_label.grid(row=3, column=0, sticky="W")
        ImWidth_entry.grid(row=3, column=1, ipadx=settingEntryLength)
        ImWidth_entry.insert(END, '1')

        ImHeight_label.grid(row=4, column=0, sticky="W")
        ImHeight_entry.grid(row=4, column=1, ipadx=settingEntryLength)
        ImHeight_entry.insert(END, '2')

        ImChannel_label.grid(row=5, column=0, sticky="W")
        ImChannel_entry.grid(row=5, column=1, ipadx=settingEntryLength)
        ImChannel_entry.insert(END, '3')

        BatchSize_label.grid(row=7, column=0, sticky="W")
        BatchSize_entry.grid(row=7, column=1, ipadx=settingEntryLength)
        BatchSize_entry.insert(END, '1')

        InitialLearningRate_label.grid(row=8, column=0, sticky="W")
        InitialLearningRate_entry.grid(row=8, column=1, ipadx=settingEntryLength)
        InitialLearningRate_entry.insert(END, '01')

        Momentum_label.grid(row=9, column=0, sticky="W")
        Momentum_entry.grid(row=9, column=1, ipadx=settingEntryLength)
        Momentum_entry.insert(END, '9')

        NumEpochs_label.grid(row=11, column=0, sticky="W")
        NumEpochs_entry.grid(row=11, column=1, ipadx=settingEntryLength)
        NumEpochs_entry.insert(END, '1')

        ChangeLearningRateEpochs_label.grid(row=12, column=0, sticky="W")
        ChangeLearningRateEpochs_format.grid(row=12, column=2, sticky="W")
        ChangeLearningRateEpochs_entry.grid(row=12, column=1, ipadx=settingEntryLength)
        ChangeLearningRateEpochs_entry.insert(END, "0, 0")

        lr_change_label.grid(row=13, column=0, sticky="W")
        lr_change_format.grid(row=13, column=2, sticky="W")
        lr_change_entry.grid(row=13, column=1, ipadx=settingEntryLength)
        lr_change_entry.insert(END, "0,0")

        WeightPrior_label.grid(row=14, column=0, sticky="W")
        WeightPrior_entry.grid(row=14, column=1, ipadx=settingEntryLength)
        WeightPrior_entry.insert(END, '01')
        WeightPrior_format.grid(row=14, column=2, sticky="W")

        class_penalty_label.grid(row=17, column=0, sticky="W")
        class_penalty_format.grid(row=17, column=2, sticky="W")
        class_penalty_entry.grid(row=17, column=1, ipadx=settingEntryLength)
        class_penalty_entry.insert(END, "1.0, 1.0")

        """

        Inputs of Parameters
        
        Augmentation

        """

        AugEnable_var = tk.IntVar()
        frame_Augmentation.place_forget()
        AugEnable_label = tk.Label(frame_Parameters, text='Usage of Augmentation: ', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        AugEnable_label.grid(row=18, column=0, sticky="W")
        Enable = tk.Radiobutton(frame_Parameters, text="Enable", variable=AugEnable_var, value=1, bg=backgroundColor,
                                indicatoron=False, width=8,
                                command=lambda: frame_Augmentation.place(rely=0.47, relx=0.02, height=300, width=600))
        Enable.grid(row=18, column=1)
        Disable = tk.Radiobutton(frame_Parameters, text="Disable", variable=AugEnable_var, value=2, bg=backgroundColor,
                                 indicatoron=False, width=8, command=lambda: frame_Augmentation.place_forget())
        Disable.grid(row=18, column=2)

        AugmentationPercentage_var = tk.IntVar()
        Rotation_var = tk.IntVar()
        Mirror_var = tk.StringVar()
        BrightnessVariation_var = tk.IntVar()
        BrightnessVariationSpot_var = tk.IntVar()
        CropPercentage_var = tk.StringVar()
        CropPixel_var = tk.StringVar()
        RotationRange_var = tk.IntVar()
        IgnoreDirection_var = tk.StringVar()
        ClassIDsNoOrientationExist_var = tk.StringVar()
        ClassIDsNoOrientation_var = tk.StringVar()

        AugmentationPercentage_label = tk.Label(frame_Augmentation, text='Percentage:',
                                                font=('calibre', 10, 'bold'),
                                                bg=backgroundColor)
        AugmentationPercentage_entry = tk.Entry(frame_Augmentation, textvariable=AugmentationPercentage_var,
                                                font=('calibre', 10, 'normal'))
        AugmentationPercentage_format = tk.Label(frame_Augmentation, text='0 to 100',
                                                 font=('calibre', 10, 'bold'),
                                                 bg=backgroundColor, fg='dark red')

        Rotation_label = tk.Label(frame_Augmentation, text='Rotation:',
                                  font=('calibre', 10, 'bold'),
                                  bg=backgroundColor)
        Rotation_entry = tk.Entry(frame_Augmentation, textvariable=Rotation_var,
                                  font=('calibre', 10, 'normal'))
        Rotation_format = tk.Label(frame_Augmentation, text='-180, -90, 0, 90, 180',
                                   font=('calibre', 10, 'bold'),
                                   bg=backgroundColor, fg='dark red')

        mirror_label = tk.Label(frame_Augmentation, text='Mirror:', font=('calibre', 10, 'bold'),
                                bg=backgroundColor)
        mirror_entry = tk.Entry(frame_Augmentation, textvariable=Mirror_var, font=('calibre', 10, 'normal'))
        mirror_format = tk.Label(frame_Augmentation, text="'r' , 'c' , 'rc' , 'off' ",
                                 font=('calibre', 10, 'bold'),
                                 bg=backgroundColor, fg='dark red')

        BrightnessVariation_label = tk.Label(frame_Augmentation, text='Brightness Range:',
                                             font=('calibre', 10, 'bold'), bg=backgroundColor)
        BrightnessVariation_entry = tk.Entry(frame_Augmentation, textvariable=BrightnessVariation_var,
                                             font=('calibre', 10, 'normal'))
        BrightnessVariation_format = tk.Label(frame_Augmentation, text="-value / +value",
                                              font=('calibre', 10, 'bold'),
                                              bg=backgroundColor, fg='dark red')

        BrightnessVariationSpot_label = tk.Label(frame_Augmentation, text='Brightness Range Focus:',
                                                 font=('calibre', 10, 'bold'),
                                                 bg=backgroundColor)
        BrightnessVariationSpot_entry = tk.Entry(frame_Augmentation, textvariable=BrightnessVariationSpot_var,
                                                 font=('calibre', 10, 'normal'), state='disabled')

        CropPercentage_label = tk.Label(frame_Augmentation, text='Crop Percentage:', font=('calibre', 10, 'bold'),
                                        bg=backgroundColor)
        CropPercentage_entry = tk.Entry(frame_Augmentation, textvariable=CropPercentage_var,
                                        font=('calibre', 10, 'normal'), state='disabled')

        CropPixel_label = tk.Label(frame_Augmentation, text='Crop Pixel:', font=('calibre', 10, 'bold'),
                                   bg=backgroundColor)
        CropPixel_entry = tk.Entry(frame_Augmentation, textvariable=CropPixel_var, font=('calibre', 10, 'normal'),
                                   state='disabled')

        RotationRange_label = tk.Label(frame_Augmentation, text='Rotation Range:',
                                       font=('calibre', 10, 'bold'), bg=backgroundColor)
        RotationRange_entry = tk.Entry(frame_Augmentation, textvariable=RotationRange_var,
                                       font=('calibre', 10, 'normal'), state='disabled')

        IgnoreDirection_label = tk.Label(frame_Augmentation, text='Ignore Direction:',
                                         font=('calibre', 10, 'bold'), bg=backgroundColor)
        IgnoreDirection_entry = tk.Entry(frame_Augmentation, textvariable=IgnoreDirection_var,
                                         font=('calibre', 10, 'normal'), state='disabled')

        ClassIDsNoOrientationExist_label = tk.Label(frame_Augmentation, text='Class IDs No Orientation Exist:',
                                                    font=('calibre', 10, 'bold'), bg=backgroundColor)
        ClassIDsNoOrientationExist_entry = tk.Entry(frame_Augmentation, textvariable=ClassIDsNoOrientationExist_var,
                                                    font=('calibre', 10, 'normal'), state='disabled')

        ClassIDsNoOrientation_label = tk.Label(frame_Augmentation, text='Class IDs No Orientation:',
                                               font=('calibre', 10, 'bold'), bg=backgroundColor)
        ClassIDsNoOrientation_entry = tk.Entry(frame_Augmentation, textvariable=ClassIDsNoOrientation_var,
                                               font=('calibre', 10, 'normal'), state='disabled')

        AugmentationPercentage_label.grid(row=0, column=0, sticky="W")
        AugmentationPercentage_entry.grid(row=0, column=1, ipadx=settingEntryLength)
        AugmentationPercentage_entry.insert(0, '0')
        AugmentationPercentage_format.grid(row=0, column=2, sticky="W")

        Rotation_label.grid(row=1, column=0, sticky="W")
        Rotation_entry.grid(row=1, column=1, ipadx=settingEntryLength)
        Rotation_entry.insert(0, '0')
        Rotation_format.grid(row=1, column=2, sticky="W")

        mirror_label.grid(row=2, column=0, sticky="W")
        mirror_entry.grid(row=2, column=1, ipadx=settingEntryLength)
        mirror_entry.insert(0, 'off')
        mirror_format.grid(row=2, column=2, sticky="W")

        BrightnessVariation_label.grid(row=3, column=0, sticky="W")
        BrightnessVariation_entry.grid(row=3, column=1, ipadx=settingEntryLength)
        BrightnessVariation_entry.insert(0, '0')
        BrightnessVariation_format.grid(row=3, column=2, sticky="W")

        BrightnessVariationSpot_label.grid(row=4, column=0, sticky="W")
        BrightnessVariationSpot_entry.grid(row=4, column=1, ipadx=settingEntryLength)
        # BrightnessVariationSpot_entry.insert(0, '0')

        CropPercentage_label.grid(row=5, column=0, sticky="W")
        CropPercentage_entry.grid(row=5, column=1, ipadx=settingEntryLength)
        # CropPercentage_entry.insert(0,'50')

        CropPixel_label.grid(row=6, column=0, sticky="W")
        CropPixel_entry.grid(row=6, column=1, ipadx=settingEntryLength)
        # CropPixel_entry.insert(0, '100')

        RotationRange_label.grid(row=7, column=0, sticky="W")
        RotationRange_entry.grid(row=7, column=1, ipadx=settingEntryLength)
        # RotationRange_entry.insert(0, '0')

        IgnoreDirection_label.grid(row=8, column=0, sticky="W")
        IgnoreDirection_entry.grid(row=8, column=1, ipadx=settingEntryLength)
        # IgnoreDirection_entry.insert(0, 'false')

        ClassIDsNoOrientationExist_label.grid(row=9, column=0, sticky="W")
        ClassIDsNoOrientationExist_entry.grid(row=9, column=1, ipadx=settingEntryLength)
        # ClassIDsNoOrientationExist_entry.insert(0, 'false')

        ClassIDsNoOrientation_label.grid(row=10, column=0, sticky="W")
        ClassIDsNoOrientation_entry.grid(row=10, column=1, ipadx=settingEntryLength)

        """
        
        Runtime, 1=CPU, 2=GPU
        
        """

        Runtime_var = tk.IntVar()
        R1 = tk.Radiobutton(frameSettings, text="Use CPU", variable=Runtime_var, value=1, bg=backgroundColor)
        R1.grid(row=0, column=0)
        R2 = tk.Radiobutton(frameSettings, text="Use GPU", variable=Runtime_var, value=2, bg=backgroundColor)
        R2.grid(row=0, column=1)

        """

        Folder Directories
        
        Directory path
        
        directory will be stored in sem-global variable *_path
        
        and used in defining operational variables

        """

        ImageDirDirectory_path = StringVar()
        PreprocessDirDirectory_path = StringVar()
        ModelDirDirectory_path = StringVar()

        def ImageDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            ImageDirDirectory_path.set(path)
            ImageDirGotLabel.config(text=path)

        ImageDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'),
                                    bg=("%s" % dir_color))
        ImageDirGotLabel.grid(row=1, column=1)
        ImageDirButton = tk.Button(frameSettings, text="Image Directory", command=lambda: ImageDirDirectory())
        ImageDirButton.grid(row=1, column=0)

        def PreprocessDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            PreprocessDirDirectory_path.set(path)
            PreprocessDirGotLabel.config(text=path)

        PreprocessDirButton = tk.Button(frameSettings, text="Preprocess Directory",
                                        command=lambda: PreprocessDirDirectory())
        PreprocessDirButton.grid(row=2, column=0)
        PreprocessDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'), bg=dir_color)
        PreprocessDirGotLabel.grid(row=2, column=1)

        def ModelDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            ModelDirDirectory_path.set(path)
            ModelDirGotLabel.config(text=path)

        ModelDirButton = tk.Button(frameSettings, text="Model Directory", command=lambda: ModelDirDirectory())
        ModelDirButton.grid(row=3, column=0)
        ModelDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'), bg=dir_color)
        ModelDirGotLabel.grid(row=3, column=1)

        """
        
        Drop List for Pretrained Models

        """

        PretrainedModelList = ["classifier_enhanced", "classifier_compact"]
        PretrainedModelList_variable = tk.StringVar()
        PretrainedModelList_variable.set(PretrainedModelList[0])
        dropList = tk.OptionMenu(frameSettings, PretrainedModelList_variable, *PretrainedModelList)

        dropList.config(width=20, font=('calibre', 10, 'bold'), bg=dir_color)
        dropList.grid(row=5, column=1)
        dropListLabel = tk.Label(frameSettings, text="Pretrained model: ", font=('calibre', 10, 'bold'),
                                 bg=backgroundColor)
        dropListLabel.grid(row=5, column=0)

        """

        Labels for Matrix

        """

        mean_precision_label = tk.Label(frame_Inspection_stats,
                                        font=('calibre', 10, 'bold'), bg=backgroundColor, fg='blue4')
        mean_recall_label = tk.Label(frame_Inspection_stats,
                                     font=('calibre', 10, 'bold'), bg=backgroundColor, fg='blue4')
        mean_f_score_label = tk.Label(frame_Inspection_stats,
                                      font=('calibre', 10, 'bold'), bg=backgroundColor, fg='blue4')

        mean_precision_label.pack(side=tk.TOP)
        mean_recall_label.pack(side=tk.TOP)
        mean_f_score_label.pack(side=tk.TOP)

        """

        Progress Indication

        """
        preProcessingText = "Pre-Process is on going..."
        preProcessingStopText = "Pre-Process Finished!"
        evaluationText = "Evaluating Results..."
        pbar = Progressbar(frameBot, mode='indeterminate', length=100)
        pre_processingLabel = tk.Label(frameBot, font=('calibre', 10, 'bold'), bg=backgroundColor)
        pre_processingLabel.place(relx=0.81)

        """

        Methods 

        """

        def startAugmentation():
            AugmentationPercentage = int(AugmentationPercentage_entry.get())
            Rotation = int(Rotation_entry.get())
            Mirror = str(mirror_entry.get())
            BrightnessVariation = int(BrightnessVariation_entry.get())
            # BrightnessVariationSpot = int(BrightnessVariationSpot_entry.get())
            # CropPercentage = int(CropPercentage_entry.get())
            # CropPixel = int(CropPixel_entry.get())
            # RotationRange = int(RotationRange_entry.get())
            # IgnoreDirection = IgnoreDirection_entry.get()
            # ClassIDsNoOrientationExist = ClassIDsNoOrientationExist_entry.get()
            # ClassIDsNoOrientationList = ClassIDsNoOrientation_entry.get().split(',')
            # ClassIDsNoOrientation = [int(i) for i in ClassIDsNoOrientationList]

            aug_call.set_input_control_param_by_name('AugmentationPercentage', AugmentationPercentage)
            aug_call.set_input_control_param_by_name('Rotation', Rotation)
            aug_call.set_input_control_param_by_name('Mirror', Mirror)
            aug_call.set_input_control_param_by_name('BrightnessVariation', BrightnessVariation)
            aug_call.set_input_control_param_by_name('BrightnessVariationSpot', 0)
            aug_call.set_input_control_param_by_name('CropPercentage', 'off')
            aug_call.set_input_control_param_by_name('CropPixel', 'off')
            aug_call.set_input_control_param_by_name('RotationRange', 0)
            aug_call.set_input_control_param_by_name('IgnoreDirection', 'false')
            aug_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', 'false')
            aug_call.set_input_control_param_by_name('ClassIDsNoOrientation', [])

            aug_call.execute()

        def startPreproc():

            AugEnable = AugEnable_var.get()

            if AugEnable == 2:
                aug_call.set_input_control_param_by_name('AugmentationPercentage', 0)
                aug_call.set_input_control_param_by_name('Rotation', 0)
                aug_call.set_input_control_param_by_name('Mirror', 'off')
                aug_call.set_input_control_param_by_name('BrightnessVariation', 0)
                aug_call.set_input_control_param_by_name('BrightnessVariationSpot', 0)
                aug_call.set_input_control_param_by_name('CropPercentage', 'off')
                aug_call.set_input_control_param_by_name('CropPixel', 'off')
                aug_call.set_input_control_param_by_name('RotationRange', 0)
                aug_call.set_input_control_param_by_name('IgnoreDirection', 'false')
                aug_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', 'false')
                aug_call.set_input_control_param_by_name('ClassIDsNoOrientation', [])
                aug_call.execute()

            elif AugEnable == 1:
                startAugmentation()

            GenParamName_augment = aug_call.get_output_control_param_by_name('GenParamName_augment')
            GenParamValue_augment = aug_call.get_output_control_param_by_name('GenParamValue_augment')

            ImageDir = ImageDirDirectory_path.get()
            PreprocessDir = PreprocessDirDirectory_path.get()
            ModelDir = ModelDirDirectory_path.get()

            # Image parameters
            ImWidth = int(ImWidth_var.get())
            ImHeight = int(ImHeight_var.get())
            ImChannel = int(ImChannel_var.get())

            # Pretrained model selection
            PretrainedModel = 'pretrained_dl_' + PretrainedModelList_variable.get() + '.hdl'

            # DL Stats setting
            BatchSize = BatchSize_entry.get()
            InitialLearningRate = InitialLearningRate_entry.get()
            Momentum = Momentum_entry.get()

            # runtime, CPU or GPU
            try:
                if Runtime_var.get() == 1:
                    Runtime = 'cpu'
                elif Runtime_var.get() == 2:
                    Runtime = 'gpu'
            except:
                messagebox.showerror("Error", "Please indicate CPU or GPU!")

            NumEpochs = NumEpochs_entry.get()

            ChangeLearningRateEpochsList = ChangeLearningRateEpochs_entry.get().split(',')
            if ChangeLearningRateEpochsList:
                try:
                    ChangeLearningRateEpochs = [int(i) for i in ChangeLearningRateEpochsList]
                except:

                    messagebox.showerror("Error", "Pleas key in correct form of ChangeLearningRateEpochs!\ne.g. 1,2,3")

            lr_changeList = lr_change_entry.get().split(',')
            if lr_changeList:
                try:
                    lr_change = [float(i) for i in lr_changeList]
                except:
                    messagebox.showerror("Error", "Pleas key in correct form of lr_change!\ne.g. 1,2,3")

            WeightPrior = WeightPrior_entry.get()
            class_penaltyList = class_penalty_entry.get().split(',')
            Class_Penalty = [float(i) for i in class_penaltyList]

            preprocess_call.set_input_control_param_by_name('RawImageBaseFolder', ImageDir)
            preprocess_call.set_input_control_param_by_name('ExampleDataDir', PreprocessDir)
            preprocess_call.set_input_control_param_by_name('BestModelBaseName',
                                                            os.path.join(ModelDir, 'best_dl_model_classification'))
            preprocess_call.set_input_control_param_by_name('FinalModelBaseName',
                                                            os.path.join(ModelDir, 'final_dl_model_classification'))
            preprocess_call.set_input_control_param_by_name('ImageWidth', ImWidth)
            preprocess_call.set_input_control_param_by_name('ImageHeight', ImHeight)
            preprocess_call.set_input_control_param_by_name('ImageNumChannels', ImChannel)
            preprocess_call.set_input_control_param_by_name('ModelFileName', PretrainedModel)
            preprocess_call.set_input_control_param_by_name('BatchSize', int(BatchSize))
            preprocess_call.set_input_control_param_by_name('InitialLearningRate', float(InitialLearningRate))
            preprocess_call.set_input_control_param_by_name('Momentum', float(Momentum))
            preprocess_call.set_input_control_param_by_name('DLDeviceType', Runtime)
            preprocess_call.set_input_control_param_by_name('NumEpochs', int(NumEpochs))
            preprocess_call.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)
            preprocess_call.set_input_control_param_by_name('lr_change', lr_change)
            preprocess_call.set_input_control_param_by_name('WeightPrior', float(WeightPrior))
            preprocess_call.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
            preprocess_call.set_input_control_param_by_name('Class_Penalty', Class_Penalty)
            preprocess_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
            preprocess_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)

            preprocess_call.execute()

        def startTraining():
            DLDataset = preprocess_call.get_output_control_param_by_name('DLDataset')
            DLModelHandle = preprocess_call.get_output_control_param_by_name('DLModelHandle')
            TrainParam = preprocess_call.get_output_control_param_by_name('TrainParam')

            training_call.set_input_control_param_by_name('DLDataset', DLDataset)
            training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
            training_call.set_input_control_param_by_name('TrainParam', TrainParam)
            training_call.set_input_control_param_by_name('StartEpoch', 0)
            training_call.set_input_control_param_by_name('Display_Ctrl', 1)

            training_call.execute()

        def startEvaluation():
            EvalBatchSize = 1
            ImWidth = int(ImWidth_var.get())
            ImHeight = int(ImHeight_var.get())
            evaluation_call.set_input_control_param_by_name('BatchSize', EvalBatchSize)
            evaluation_call.set_input_control_param_by_name('ModelDir', ModelDirDirectory_path.get())
            evaluation_call.set_input_control_param_by_name('ExampleDataDir', PreprocessDirDirectory_path.get())
            evaluation_call.set_input_control_param_by_name('ImageWidth', ImWidth)
            evaluation_call.set_input_control_param_by_name('ImageHeight', ImHeight)
            evaluation_call.execute()

            output_EvalResults = evaluation_call.get_output_control_param_by_name('EvaluationResult')
            confusion_matrix_tuple = ha.get_dict_tuple(output_EvalResults, 'absolute_confusion_matrix')
            confusion_matrix_List = ha.get_full_matrix(confusion_matrix_tuple)
            values_inside_global = ha.get_dict_tuple(output_EvalResults, 'global')
            mean_precision = ha.get_dict_tuple(values_inside_global, 'mean_precision')
            mean_recall = ha.get_dict_tuple(values_inside_global, 'mean_recall')
            mean_f_score = ha.get_dict_tuple(values_inside_global, 'mean_f_score')

            listout = [confusion_matrix_List, mean_precision, mean_recall, mean_f_score]

            return listout

        def preprocess_Run():

            pbar.place(relx=0.93)
            pbar.start()
            proc1 = threading.Thread(target=startPreproc)
            proc1.start()
            while proc1.is_alive():
                pre_processingLabel.config(text=preProcessingText)
                messagebox.showinfo("Pre-Processing", "Pre-Processing is on going...")

            while not proc1.is_alive():
                pbar.stop()
                pbar.place_forget()
                pre_processingLabel.config(text=preProcessingStopText)

                messagebox.showinfo('Finished', "Pre-Processing finished, please proceed.")
                proc1.join()
                break

        def training_Run():

            proc2 = threading.Thread(target=startTraining)
            proc2.start()
            if proc2.is_alive():
                messagebox.showerror('Training', "Training ongoing, please view the graphic window.")
            if not proc2.is_alive():
                messagebox.showinfo('Finished', "Training finished, thank you.")
                proc2.join()
            print(threading.active_count())

        def evalutation_Run():
            pbar.place(relx=0.93)
            pbar.start()
            pre_processingLabel.config(text='Evaluating Results...')
            proc3 = threading.Thread(target=startEvaluation)
            pbar.place(relx=0.93)
            proc3.start()

            while proc3.is_alive():
                pre_processingLabel.config(text=evaluationText)
                messagebox.showinfo("Evaluation", 'Evalutation Running...')
            while not proc3.is_alive():
                pbar.stop()
                pbar.place_forget()
                pre_processingLabel.config(text='Evaluation Done!')
                proc3.join()
                break

            listFromEvaluation = startEvaluation()
            confusion_matrix_List = listFromEvaluation[0]
            mean_precision = listFromEvaluation[1]
            mean_recall = listFromEvaluation[2]
            mean_f_score = listFromEvaluation[3]

            labels = []
            categoriesDir = ImageDirDirectory_path.get() + '/Train/'
            categoriesRaw = [y[0] for y in os.walk(categoriesDir)]
            categoriesRaw.remove(categoriesRaw[0])
            categories = []
            for i in range(len(categoriesRaw)):
                categories.append(os.path.basename(categoriesRaw[i]))

            for j in range(len(categories)):
                Prediction = 'Prediction:' + categories[j] + '\n'
                for k in range(len(categories)):
                    labelString = Prediction + 'Truth: ' + categories[k]
                    labels.append(labelString)

            print(startEvaluation())
            print(confusion_matrix_List)

            lengthCM = numpy.sqrt(len(confusion_matrix_List))
            confusion_matrix_List_Chunks = list(divide_chunks(confusion_matrix_List, int(lengthCM)))
            confusion_matrix_Array = numpy.asarray(confusion_matrix_List_Chunks)
            make_confusion_matrix(confusion_matrix_Array,
                                  group_names=labels,
                                  categories=categories,
                                  cmap='Blues',
                                  xyplotlabels=True,
                                  cbar=False)
            # sns.heatmap(matrix, square=True, cbar=False, ax=ax,annot=True)
            # sns.heatmap(cf_mat_val, annot=True,
            #            fmt='.2%', cmap='Blues', ax=ax)
            print(confusion_matrix_Array)
            canvas = FigureCanvasTkAgg(figure, master=frame_Inspection_graphical)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            mean_precision_label.config(text='Mean Precision:\n' + str(mean_precision) + '\n\n', )
            mean_recall_label.config(text='Mean Recall:\n' + str(mean_recall) + '\n\n')
            mean_f_score_label.config(text='Mean F1 Score:\n' + str(mean_f_score) + '\n\n')

        def stop():
            """Enable scanning by setting the global flag to True."""
            global running
            running = True
            FileHandle = ha.open_file('mutex.dat', 'output')

            ha.fwrite_string(FileHandle, 1)
            # close_file(FileHandle)

        stopbutton = ttk.Button(self, text="stop  ", command=lambda: stop())
        stopbutton.place(rely=0.01, relx=0.4, height=30, width=30)


        def control_execution():
            idx = 0  # loop index

            def stop():
                """Enable scanning by setting the global flag to True."""
                global running
                running = True
                FileHandle = ha.open_file('mutex.dat', 'output')

                ha.fwrite_string(FileHandle, 1)
                # close_file(FileHandle)

            def resume():
                """Stop scanning by setting the global flag to False."""
                global running
                running = False
                FileHandle = ha.open_file('mutex.dat', 'output')
                ha.fwrite_string(FileHandle, 0)
                # close_file(FileHandle)

            def doSomething():
                root.destroy()

            def on_exit():
                """When you click to exit, this function is called"""
                if messagebox.askyesno("Exit", "Do you want to quit the application?"):
                    root.destroy()
                    FileHandle = ha.open_file('mutex.dat', 'output')
                    ha.fwrite_string(FileHandle, 3)

            root = Tk.Tk()
            root.title("Contol execution")
            root.protocol('WM_DELETE_WINDOW', on_exit)  # root is your root window
            # root.geometry('200x250 + 400 + 300')

            start = Tk.Button(root, text="Stop", command=stop)
            stop = Tk.Button(root, text="Resume", command=resume)

            start.grid()
            stop.grid()

            # open_file(), fwrite_line() and close_file() to modify the 'mutex.dat'

            while True:
                if idx % 500 == 0:
                    root.update()

                if running:
                    idx += 1

                else:
                    idx += 1