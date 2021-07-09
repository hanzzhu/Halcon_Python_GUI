# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:04:03 2020

@author: 683955
"""

import os
import halcon as ha

# Input Control Parameters
ImageDir = 'D:/930415/TE_consolidate/Consolidate'
PreprocessDir = 'D:/930415/TE_consolidate/split'
ModelDir = 'D:/930415/TE_consolidate/model'

ImWidth = 100
ImHeight = 100
ImChannel = 3

PretrainedModel = 'pretrained_dl_classifier_compact.hdl'

BatchSize = 1
InitialLearningRate = 1e-3
Momentum = 0.9
Runtime = 'gpu'
NumEpochs = 1
ChangeLearningRateEpochs = ()
lr_change = ()
WeightPrior = 1e-3
aug_percent = 80
mirror_image = 'c'
class_penalty = (1.0, 1.0)
eval_result = {}

AugmentationPercentage = 0
Rotation = 0
Mirror = 'off'
BrightnessVariation = 0
BrightnessVariationSpot = 0
CropPercentage = 'off'
CropPixel = 'off'
RotationRange = 0
IgnoreDirection = 'false'
ClassIDsNoOrientationExist = 'false'
ClassIDsNoOrientation = []

engine = ha.HDevEngine()
engine.set_procedure_path('C:/MVTec/Halcon-20.11-Progress/procedures')
engine.set_procedure_path('D:/930415')  # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located
program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')

aug = ha.HDevProcedure.load_local(program, 'augment_prepare')
aug_call = ha.HDevProcedureCall(aug)

aug_call.set_input_control_param_by_name('AugmentationPercentage', AugmentationPercentage)
aug_call.set_input_control_param_by_name('Rotation', Rotation)
aug_call.set_input_control_param_by_name('Mirror', Mirror)
aug_call.set_input_control_param_by_name('BrightnessVariation', BrightnessVariation)
aug_call.set_input_control_param_by_name('BrightnessVariationSpot', BrightnessVariationSpot)
aug_call.set_input_control_param_by_name('CropPercentage', CropPercentage)
aug_call.set_input_control_param_by_name('CropPixel', CropPixel)
aug_call.set_input_control_param_by_name('RotationRange', RotationRange)
aug_call.set_input_control_param_by_name('IgnoreDirection', IgnoreDirection)
aug_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', ClassIDsNoOrientationExist)
aug_call.set_input_control_param_by_name('ClassIDsNoOrientation', ClassIDsNoOrientation)

aug_call.execute()

GenParamName_augment = aug_call.get_output_control_param_by_name('GenParamName_augment')
GenParamValue_augment = aug_call.get_output_control_param_by_name('GenParamValue_augment')

proc1 = ha.HDevProcedure.load_local(program, 'prepare_for_training')
proc_call1 = ha.HDevProcedureCall(proc1)

proc_call1.set_input_control_param_by_name('RawImageBaseFolder', ImageDir)
proc_call1.set_input_control_param_by_name('ExampleDataDir', PreprocessDir)
proc_call1.set_input_control_param_by_name('BestModelBaseName', os.path.join(ModelDir, 'best_dl_model_classification'))
proc_call1.set_input_control_param_by_name('FinalModelBaseName',
                                           os.path.join(ModelDir, 'final_dl_model_classification'))

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
proc_call1.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
proc_call1.set_input_control_param_by_name('Class_Penalty', class_penalty)
proc_call1.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
proc_call1.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)

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

# x = proc_call.get_output_control_param_by_name('EvaluationResult')[0]

proc3 = ha.HDevProcedure.load_local(program, 'Evaluation')
proc_call3 = ha.HDevProcedureCall(proc3)

EvalBatchSize = 10
proc_call3.set_input_control_param_by_name('BatchSize', EvalBatchSize)

proc_call3.set_input_control_param_by_name('ModelDir', ModelDir)
proc_call3.set_input_control_param_by_name('ExampleDataDir', PreprocessDir)
proc_call3.set_input_control_param_by_name('ImageWidth', ImWidth)
proc_call3.set_input_control_param_by_name('ImageHeight', ImHeight)
proc_call3.execute()

output_EvalResults = proc_call3.get_output_control_param_by_name('EvaluationResult')
cf_mat = ha.get_dict_tuple(output_EvalResults, 'absolute_confusion_matrix')
cf_mat_val = ha.get_full_matrix(cf_mat)

metric = ha.get_dict_tuple(output_EvalResults, 'global')
mp = ha.get_dict_tuple(metric, 'mean_precision')
