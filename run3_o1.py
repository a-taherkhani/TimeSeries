# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:06:13 2018

@author: ERD204
"""

#from main_AT_laptop_perm4_twoOut10 import build_model
from main_o1 import build_model

import os
#se the path to your data set
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)# Get the directory name of the script

 
classifier_name='resnet_AT5_base_twoIn6_11'#   

################uni variant:
archive_name = 'UCR_TS_Archive_2015'#'mts_archive'#'UCR_TS_Archive_2015'

dataset_names = ['RandTS']
###########multui variant:
acc={}#out1
acc_last={}
precision_last={}
recall_last= {}
f1_last= {}

acc_lastMax ={}
acc_lastMoxVot ={}

Elaps_times = {}


w_p=[0.5,0.5, 1]#parametter for 



for dataset_name in dataset_names:
    ###loda trained model:
    load_model_dir = root_dir+'/results/1500resnet_AT5/UCR_TS_Archive_2015_itr_8/'+dataset_name+'/best_model.hdf5'

#############################################   
    accuracy, precision_s, recall_s, f1_s, accuracy_onMax, Elaps_time = build_model(dataset_name, w_p, archive_name,classifier_name, root_dir)#single out 


    acc_last[dataset_name]=accuracy
    
    precision_last[dataset_name]=precision_s
    recall_last[dataset_name]=recall_s
    f1_last[dataset_name]=f1_s    
    
    acc_lastMax[dataset_name] = accuracy_onMax
    
    Elaps_times [dataset_name] = Elaps_time
    
    
    print('acc_lastMax')

    print ( acc_lastMax)
    
    print('##########acc_last###########')

    print ( acc_last)
    
    print('precision:')
    print(precision_last)    
    
        
    print(' #########recall##########:')
    print( recall_last)    
    
        
    print(' F1:')
    print( f1_last)    
    
    
    print('Elaps_times:')
    print (Elaps_times)
    ##############################################
    
    
    
#print (acc)
#import main_AT_CPU_perm4_twoOut4t
#main_AT_CPU_perm4_twoOut4t.Run()
