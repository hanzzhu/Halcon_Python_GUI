[HalconGUI Document.docx](https://github.com/Hanzzzzzzzz/HalconPythonGUI/files/6162224/HalconGUI.Document.docx)
# HalconPythonGUI

This is a GUI built from tkinter.

# Motivation
In this project, I would like to provide centralised user expeirence for conducting Deep Learning tasks with the help of Halcon.
Users could operate Halcon without opening Halcon software, which is time-consuming to learn the basics.

# About the Project
In this repo, Halcon related codes were deleted for privacy purposes.
The full structured GUI should include I/O controllers to retrieve data from Halcon, with commands operating Halcon procedures. Therefore, Hdev files should be written first in order fo the GUI to run.
This repo is a basic showcase of GUI, i.e page layouts, operations etc.

To start, run main.py. There is exe file in /dist, however an outdated version.

# Contribution
Suggestions and enqueries are welcome!

Introduction 
Created base on Python TKinter and Halcon. The app first intake user’s inputs as parameters, then call and operate respective Halcon procedures and commands for DL.
Ultimate aim is to achieve classification, object detection and anomaly detection functions within the single GUI.
Halcon procedure files should be prepared in the first place, as these Halcon codes are the working core. 
There are 3 major steps being executed by Halcon: pre-process, training, and evaluation. 
In this document, ClassificationPage.py will be explained in detail with natural sequence of the codes.

Detailed Explanation

Setting Up
1.	Initialize Halcon by setting HDevEngine. 
2.	Set path for Halcon procedure library. Note that other than the mandatory path of default Halcon procedures, path where newest DL library files (filenames end with _PK.hdpl) are located should also be set.
3.	Set path for the exact Halcon program(.hdev) we are using. 
program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')

Then, set respective Halcon procedures in the program, which will be used later. E.g. augment_prepare. 
aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))

 
(Procedures inside the Halcon program)

4.	Note that we could skip setting Halcon program and use the procedures directly from Halcon libraries. E.g.
augment_proc_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('augment_prepare'))
  
GUI Layout
1.	Frames. Create several individual frames to separate segments. 

2.	Operational buttons. 
 
Each button is linked to dedicated method. Operational sequence is from left to right, as ruled in Halcon program.

3.	Parameter entry. User key in values that control the DL process. Augmentation segment is being controlled by a switch.
   

4.	Directory, runtime and pretrained model selection. Note that for classification, three directories will be needed.
  

 
5.	Progress Indication. Progress bar and labels showing the current condition of pre-process will be shown bottom right corner.
         

6.	Evaluation. After training, confusion matrix and important training data will be shown.



Methods
In this class, there are 2 types of methods.
1.	Operational methods that interact with Halcon commands. 		               startAugmentation(), startPreproc(), startTraining(), startEvaluation().

2.	Methods that perform respective operational methods with threading. 	              preprocess_Run(), training_Run(),evalutation_Run().

startAugmentation() : Using user inputs to call Halcon procedure augment_prepare(). Some outputs will be used in pre-process.
startPreproc() : Pre-process. Inside this method, augmentation is executed. Using the outputs from augmentation and more other relative user inputs, Halcon procedure in prepare_for_training() procedure will be called.  Some outputs will be used in training.
startTraining():Training. Using outputs from preprocess to conduct DL training. Will form trained models in selected folder.
startEvaluation(): Evaluation. Results could be extracted from Hhandles as python tuple, and then make lists for confusion matrix.

preprocess_Run() & training_Run(): Utilize multi thread to perform the tasks.
evalutation_Run(): Other than utilizing multi threads, confusion matrix will be drawn by calling makeCM class, which involves using matlibplot to draw the matrix and showing labels. 





