# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:04:03 2020

@author: 683955
"""

import os
import halcon as ha

# Input Control Parameters
ImageDir = 'D:/930415/HK/animal'
PreprocessDir = 'D:/930415/HK/Split'
ModelDir = 'D:/930415/HK/model'

ImWidth = 100
ImHeight = 100
ImChannel = 3

PretrainedModel = 'pretrained_dl_classifier_compact.hdl'

BatchSize = 16
InitialLearningRate = 1e-3
Momentum = 0.9
Runtime = 'gpu'
NumEpochs = 6
ChangeLearningRateEpochs = ()
lr_change = ()
WeightPrior = 1e-3
aug_percent = 80
# mirror_image = 'c'
# class_penalty = (1.0,1.0)
# eval_result = {}

engine = ha.HDevEngine()
engine.set_procedure_path('C:/MVTec/Halcon-20.11-Progress/procedures')
engine.set_procedure_path('D:/930415')  # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located
    

program = ha.HDevProgram('DL_train_CL_fbw.hdev')
proc1 = ha.HDevProcedure.load_local(program, 'prepare_for_training')
proc_call1 = ha.HDevProcedureCall(proc1)

proc_call1.set_input_control_param_by_name('RawImageBaseFolder', ImageDir)
proc_call1.set_input_control_param_by_name('ExampleDataDir', PreprocessDir)
proc_call1.set_input_control_param_by_name('BestModelBaseName', os.path.join(ModelDir,'best_dl_model_classification'))
proc_call1.set_input_control_param_by_name('FinalModelBaseName', os.path.join(ModelDir,'final_dl_model_classification'))

proc_call1.set_input_control_param_by_name('ImageWidth', ImWidth)
proc_call1.set_input_control_param_by_name('ImageHeight', ImHeight)
proc_call1.set_input_control_param_by_name('ImageNumChannels', ImChannel)

proc_call1.set_input_control_param_by_name('ModelFileName', PretrainedModel)

proc_call1.set_input_control_param_by_name('BatchSize', BatchSize)
proc_call1.set_input_control_param_by_name('InitialLearningRate', InitialLearningRate)
proc_call1.set_input_control_param_by_name('Momentum', Momentum)
proc_call1.set_input_control_param_by_name('DLDeviceType', Runtime)
proc_call1.set_input_control_param_by_name('NumEpochs', NumEpochs)
proc_call1.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)
proc_call1.set_input_control_param_by_name('lr_change', lr_change)
proc_call1.set_input_control_param_by_name('WeightPrior', WeightPrior)
proc_call1.set_input_control_param_by_name('aug_percent', aug_percent)
proc_call1.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
#proc_call1.set_input_control_param_by_name('mirror_image', mirror_image)
#proc_call1.set_input_control_param_by_name('class_penalty', class_penalty)

proc_call1.execute()

proc2 = ha.HDevProcedure.load_local(program, 'train_dl_model_PK')
proc_call2 = ha.HDevProcedureCall(proc2)

DLDataset = proc_call1.get_output_control_param_by_name('DLDataset')
DLModelHandle = proc_call1.get_output_control_param_by_name('DLModelHandle')
TrainParam = proc_call1.get_output_control_param_by_name('TrainParam')

proc_call2.set_input_control_param_by_name('DLDataset', DLDataset)
proc_call2.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
proc_call2.set_input_control_param_by_name('TrainParam', TrainParam)
proc_call2.set_input_control_param_by_name('StartEpoch', 0)

proc_call2.execute()

output_TrainResults = proc_call2.get_output_control_param_by_name('TrainResults')

output_TrainInfos = proc_call2.get_output_control_param_by_name('TrainInfos')
print(str(output_TrainResults))
#x = proc_call.get_output_control_param_by_name('EvaluationResult')[0]


