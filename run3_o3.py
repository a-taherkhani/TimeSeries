# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:06:13 2018

@author: A.T
"""

from main_AT_laptop_perm4_twoOut8 import build_model

#se the path to your data set
root_dir = 'C:/Users/ataher00/OneDrive - De Montfort University/report2/python/time series'

classifier_name='resnet_AT5_base_twoIn6_10'#   # one block before intermediate target

################uni variant:
archive_name = 'UCR_TS_Archive_2015'#'mts_archive'#'UCR_TS_Archive_2015'
#dataset_names = [ 'ArrowHead', 'Beef', 'Car', 'ECG200', 'Ham', 'Herring', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MoteStrain', 'ToeSegmentation1']
dataset_names = [ 'ArrowHead', 'Beef', 'Car', 'ECG200', 'Ham']# 'Herring', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MoteStrain', 'ToeSegmentation1']

#dataset_names = ['ArrowHead']
###########multui variant:
#archive_name = 'mts_archive'#'UCR_TS_Archive_2015'
#dataset_names = [ 'cornell']
##################Machine helth monitoring:
#archive_name = 'machine_helth_data'#'mts_archive'#'UCR_TS_Archive_2015'
#dataset_names = ['dat_seq']

acc={}#out1

acc_last={}
precision_last={}
recall_last={}
f1_last={}

acc_lastMax ={}
acc_lastMoxVot ={}

Elaps_times = {}

w_p=[0.5,0.5, 1]#parametter for 


for dataset_name in dataset_names:
    ###loda trained model:
    load_model_dir = root_dir+'/results/1500resnet_AT5/UCR_TS_Archive_2015_itr_8/'+dataset_name+'/best_model.hdf5'
#############################################   
    # delet the previous results to prevent error: C:/Users/ERD204/Documents/data/time_series/results/resnet_AT5_base_twoIn6_10/UCR_TS_Archive_2015_itr_8/ArrowHead/
    acc1, accuracy,precision_s, recall_s, f1_s, accuracy_onMax, accuracy_onMaxVot, Elaps_time = build_model(dataset_name, w_p, archive_name,classifier_name, root_dir)

    #acc.append(acc1)
    acc[dataset_name]=acc1
    
    acc_last[dataset_name]=accuracy
    precision_last[dataset_name]=precision_s
    recall_last[dataset_name]=recall_s
    f1_last[dataset_name]=f1_s
    
    acc_lastMax[dataset_name] = accuracy_onMax
    acc_lastMoxVot[dataset_name] = accuracy_onMaxVot
    
    Elaps_times [dataset_name] = Elaps_time
    print('acc, acc_last, acc_lastMax, acc_lastMoxVot')
    print (acc, acc_last, acc_lastMax, acc_lastMoxVot)
    print('precision, recall, F1:')
    print(precision_last, recall_last, f1_last)
    
    print('Elaps_times:')
    print (Elaps_times)
    
