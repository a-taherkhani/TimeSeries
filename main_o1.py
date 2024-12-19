#############
from keras import backend as K
K.clear_session()
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
#######################
from utils.utils import generate_results_csv
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import pandas as pd

import numpy as np
import sys
import sklearn 
import copy

#####utils_AT
from utils_AT import read_dataset_cornell, gen_val_set2, gen_val_set, plot_confusion_matrix, data_permutations, balance_data, test_data_periparation,test_data_periparation_twoOut,  calculate_accuracy#, test_balance_data
#from utils_AT import calculate_accuracyMax, calculate_accuracyMaxVot
from utils_AT import calculate_accuracyMax3_single_o, calculate_accuracyMax3, calculate_accuracyMaxVot3 # three output
from utils_AT import save_logs_AT_o1o3, calculate_accuracy_precision


#main_AT_CPU_perm4_twoOut5.Run()
####down sampling:
from imblearn.under_sampling import NearMiss#https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html
from collections import Counter
    

def build_model(dataset_name, w_p, archive_name, classifier_name, root_dir):    #############randome seed:
    #seed = 100
    seed = 50
    np.random.seed(seed)
    
    
    
    #TensorFlow has its own random number generator

    import tensorflow
    tensorflow.random.set_seed(seed)
    
    ####################
    def data_preparation(datasets_dict, dataset_name):#mix original x_tes with train data 
    
            x_train = datasets_dict[dataset_name][0]
            y_train = datasets_dict[dataset_name][1]
            x_test = datasets_dict[dataset_name][2]
            y_test = datasets_dict[dataset_name][3]

        
            ####################################
            nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))
        
            # make the min to zero of labels
            y_train,y_test = transform_labels(y_train,y_test)
        
            # save orignal y because later we will use binary
            y_true = y_test.astype(np.int64) 
            # transform the labels from integers to one hot vectors
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
            y_train = enc.transform(y_train.reshape(-1,1)).toarray()
            y_test = enc.transform(y_test.reshape(-1,1)).toarray()
            ################################################
            for x_train_, x_test_ in zip(x_train, x_test):    
                if len(x_train_.shape) == 2: # if univariate 
                    # add a dimension to make it multivariate with one dimension 
                    x_train_ = x_train_.reshape((x_train_.shape[0],x_train_.shape[1],1))
                    x_test_ = x_test_.reshape((x_test_.shape[0],x_test_.shape[1],1))
            ###########################################################################        
            return x_test, y_test, y_train
    ########################
            
    def new_combination_data():
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        ######    ######AT data permutation
        x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        x_testt, y_testt = test_balance_data(x_testt, y_testt)
        x_test = x_testt
        y_test = y_testt
    
        x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)
        x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint)
        x_train = x_traint
        y_train = y_traint
        
        datasets_dict1 = {	dataset_name: (x_train, y_train, x_test, y_test)};
        
        
        return datasets_dict1
    
    def new_combination_data_twoIn():# make an input list that contine x1 and x2
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        ######    ######AT data permutation
        x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        ##################################
        x_test = [x_test1, x_test2]
    ############################################
        y_test = y_testt
    
        x_train1, x_train2, x_traint, y_traint = test_data_periparation(x_train, y_train, x_train, y_train)#it considered (i,i)
        ######################
        x_train = [x_train1, x_train2]
        ###########################
        y_train = y_traint
        
        datasets_dict1 = {	dataset_name: (x_train, y_train, x_test, y_test)};
    
        
        return datasets_dict1
    def new_combination_data_twoIn_twoOut(val=0):# make an input list that contine x1 and x2
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        ######    ######AT data permutation
    #    x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
        
        ##################################
        x_test = [x_test1, x_test2]
    ############################################
        y_test =[ y_test1, y_test2, y_testt ]
        x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)
    
        ######################
        x_train = [x_train1, x_train2]
        ###########################
        y_train = y_traint
        y_train =[ y_train1, y_train2, y_traint]
        ###############################
        if val==1:
            print ('randomlly select validation set' )
            x_train, y_train, x_val, y_val = gen_val_set(x_train, y_train)
            datasets_dict1 = {	dataset_name: (x_train, y_train, x_val, y_val, x_test, y_test)};
        else:
            ############################
            datasets_dict1 = {	dataset_name: (x_train, y_train, x_test, y_test)};
    
        
        return datasets_dict1
    
    def new_combination_data_twoIn_twoOut2(val=0, down_sample=0):# make an input list that contine x1 and x2
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        if val == 0:
            
            x_train = datasets_dict0[dataset_name][0]
            y_train = datasets_dict0[dataset_name][1]
            x_test = datasets_dict0[dataset_name][2]
            y_test = datasets_dict0[dataset_name][3]
            ######    ######AT data permutation
            if down_sample ==1:
                print('\n')
                print(dataset_name)
                print('before down-sampling:')
                print(sorted(Counter(y_train).items()))
                count=Counter(y_train)
                for key in count.keys():
                   count[key]=2# 2 for Mallat

                nm1 = NearMiss(sampling_strategy=count, version=1, n_neighbors=2)

                X_resampled_nm1, y_resampled = nm1.fit_resample(x_train, y_train)                
                
                
                print('after down-sampling:')
                print(sorted(Counter(y_resampled).items()))
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
            else:
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
            
        
        ###################
            x_test = [x_test1, x_test2]
            y_test =[ y_test1, y_test2, y_testt ]
            ######traind data:
            if down_sample ==1:
               x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
            else:
               x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)

 
            ######################
            x_train = [x_train1, x_train2]
            ###########################
            y_train =[ y_train1, y_train2, y_traint]
            ###############################
            if down_sample==1:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(),X_resampled_nm1.copy(), y_resampled.copy())};
                
            else:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())};
          
            
            return datasets_dict1, datasets_dict0
        
        elif val==1:
            x_train = datasets_dict0[dataset_name][0]
            y_train = datasets_dict0[dataset_name][1]
            x_test = datasets_dict0[dataset_name][2]
            y_test = datasets_dict0[dataset_name][3]
            ################        
            print ('randomlly select validation set' )
            x_train, y_train, x_val, y_val = gen_val_set2(x_train, y_train)
            datasets_dict0_val = {dataset_name:((x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy()))}
            
            ######    ######AT data permutation
            if down_sample ==1:
                ###############downsample training data:
                print('\n')
                print(dataset_name)
                print('before down-sampling:')
                print(sorted(Counter(y_train).items()))
                count=Counter(y_train)
                for key in count.keys():
                   count[key]=2# 2 for Mallat

                nm1 = NearMiss(ratio= count,version=1, n_neighbors=2)
                X_resampled_nm1, y_resampled = nm1.fit_sample(x_train, y_train)
                
                print('after down-sampling:')
                print(sorted(Counter(y_resampled).items()))
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
                ############################
            else:
                    
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
            
        ###################
            x_test = [x_test1, x_test2]
            y_test =[ y_test1, y_test2, y_testt ]
             ###########validation data:
            if down_sample == 1:
                x_val1, x_val2, x_valt, y_val1, y_val2, y_valt = test_data_periparation_twoOut(x_val, y_val,X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]

                
            else:
                
                x_val1, x_val2, x_valt, y_val1, y_val2, y_valt = test_data_periparation_twoOut(x_val, y_val, x_train, y_train)#
            x_val = [x_val1, x_val2]
            y_val =[ y_val1, y_val2, y_valt ]
            ############################################   
            if down_sample==1:
                x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]

            else:
                x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)
            
            ######################
            x_train = [x_train1, x_train2]
            ###########################
            y_train =[ y_train1, y_train2, y_traint]

                ############################
            if down_sample==1:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy(),X_resampled_nm1.copy(), y_resampled.copy())};
                
            else:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy())};
            
        
            
            return datasets_dict1, datasets_dict0_val
    
    def new_combination_data_twoIn_twoOut3(val=0, down_sample=0):# make an input list that contine x1 and x2/base network one out
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        if val == 0:
            
            x_train = datasets_dict0[dataset_name][0]
            y_train = datasets_dict0[dataset_name][1]
            x_test = datasets_dict0[dataset_name][2]
            y_test = datasets_dict0[dataset_name][3]
            ######    ######AT data permutation
            if down_sample ==1:
                print('\n')
                print(dataset_name)
                print('before down-sampling:')
                print(sorted(Counter(y_train).items()))
                count=Counter(y_train)
                for key in count.keys():
                   count[key]=2# 2 for Mallat

                nm1 = NearMiss(ratio= count,version=1, n_neighbors=2)
                
                X_resampled_nm1, y_resampled = nm1.fit_sample(x_train, y_train)
                
                print('after down-sampling:')
                print(sorted(Counter(y_resampled).items()))
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
            else:
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
            
        
        ###################
            x_test = [x_test1, x_test2]
            y_test = y_testt 
            
            ######traind data:
            if down_sample ==1:
               x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
            else:
               x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)

 
            ######################
            x_train = [x_train1, x_train2]
            ###########################
            y_train = y_traint

            ###############################
            if down_sample==1:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(),X_resampled_nm1.copy(), y_resampled.copy())};
                
            else:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())};
 
            
            return datasets_dict1, datasets_dict0
        
        elif val==1:
            x_train = datasets_dict0[dataset_name][0]
            y_train = datasets_dict0[dataset_name][1]
            x_test = datasets_dict0[dataset_name][2]
            y_test = datasets_dict0[dataset_name][3]
            ################        
            print ('randomlly select validation set' )
            x_train, y_train, x_val, y_val = gen_val_set2(x_train, y_train)
            datasets_dict0_val = {dataset_name:((x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy()))}
            
            ######    ######AT data permutation
            if down_sample ==1:
                ###############downsample training data:
                print('\n')
                print(dataset_name)
                print('before down-sampling:')
                print(sorted(Counter(y_train).items()))
                count=Counter(y_train)
                for key in count.keys():
                   count[key]=2# 2 for Mallat

                nm1 = NearMiss(ratio= count,version=1, n_neighbors=2)
                
                X_resampled_nm1, y_resampled = nm1.fit_sample(x_train, y_train)
                
                print('after down-sampling:')
                print(sorted(Counter(y_resampled).items()))
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]
                ############################
            else:
                    
                x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
            
            ##################################
            x_test = [x_test1, x_test2]
            y_test = y_testt 
            
             ###########validation data:
            if down_sample == 1:
                x_val1, x_val2, x_valt, y_val1, y_val2, y_valt = test_data_periparation_twoOut(x_val, y_val,X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]

                
            else:
                
                x_val1, x_val2, x_valt, y_val1, y_val2, y_valt = test_data_periparation_twoOut(x_val, y_val, x_train, y_train)#
            x_val = [x_val1, x_val2]
            y_val = y_valt 
            
            ############################################   
            if down_sample==1:
                x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, X_resampled_nm1, y_resampled)#[X_resampled_nm1, y_resampled]=[x_train, y_train]

            else:
                x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)
            
        
        
            ######################
            x_train = [x_train1, x_train2]
            ###########################
            y_train = y_traint

                ############################
            if down_sample==1:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy(),X_resampled_nm1.copy(), y_resampled.copy())};
                
            else:
                datasets_dict1 = {	dataset_name: (x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy())};
            
        
            
            return datasets_dict1, datasets_dict0_val    
   
    def total_test_data(train=0):# make an input list that contine [x1, x2]. 390*391
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        if train ==1:
            x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_train, y_train, x_train, y_train)#
     
        else:
                ######    ######AT data permutation
            x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        ##################################
        x_test = [x_test1, x_test2]
    ############################################
        x_test00 = []
        for i, modality in enumerate(x_test):
                x_test00.append(np.expand_dims(modality, axis=2))
    #    y_test = y_testt
        return x_test00
    def total_test_data2(train=0):# make an input list that contine [x1, x2]. 390*391
        #1)combine test data with trainig data
        #2)generat different permutation of train data and make it imbalance
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        if train ==1:
            x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_train, y_train, x_train, y_train)#
     
        else:
                ######    ######AT data permutation
            x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        ##################################
         #    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt)
        x_test = [x_test1, x_test2]
    ############################################
        x_test00 = []
        for i, modality in enumerate(x_test):
                x_test00.append(np.expand_dims(modality, axis=2))
    #    y_test = y_testt
        return x_test00, y_testt
        
    def fit_classifier(w_p): 
        x_train = copy.deepcopy(datasets_dict[dataset_name][0])
        y_train = copy.deepcopy(datasets_dict[dataset_name][1])
        x_test = copy.deepcopy(datasets_dict[dataset_name][2])#or validation
        y_test = copy.deepcopy(datasets_dict[dataset_name][3])
    #    ####################################
        nb_classes = len(np.unique(np.concatenate((y_train[0],y_test[0]),axis =0)))
    
        nb_classes_last = 2
        # make the min to zero of labels
        for i, (yTrain, yTest) in enumerate( zip(y_train, y_test)):
            y_train[i],y_test[i] = transform_labels(yTrain, yTest)
    
        # save orignal y because later we will use binary
        y_true =[]
        for i, yTrue in enumerate( y_test):
            y_true.append(yTrue.astype(np.int64) )
    #        y_true[i]=yTrue.astype(np.int64) 
        # transform the labels from integers to one hot vectors
        for i, (yTrain, yTest) in enumerate(zip(y_train, y_test)):
            enc = sklearn.preprocessing.OneHotEncoder()
            enc.fit(np.concatenate((yTrain,yTest),axis =0).reshape(-1,1))
            y_train[i] = enc.transform(yTrain.reshape(-1,1)).toarray()
            y_test[i] = enc.transform(yTest.reshape(-1,1)).toarray()
    ##################################################################
        for i, (x_train_, x_test_) in enumerate(zip(x_train, x_test)):
            if len(x_train_.shape) == 2: # if univariate 
                # add a dimension to make it multivariate with one dimension 
                x_train_ = x_train_.reshape((x_train_.shape[0],x_train_.shape[1],1))
                x_train[i] = x_train_
                x_test_ = x_test_.reshape((x_test_.shape[0],x_test_.shape[1],1))
                x_test[i]=x_test_
    ################################################################################
        input_shape = x_train[0].shape[1:]
                
        classifier = create_classifier(classifier_name,input_shape, nb_classes, nb_classes_last, output_directory, dataset_name, root_dir)
        
        #########generate Class_weights:
        from sklearn.utils import class_weight
        class_weights_dic =[]
    
        for i, yTrain in enumerate( y_train):
    
            y_train_OH =np.argmax(yTrain, axis=1)# One hot encoded vector
            
            class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes= np.unique(y_train_OH), y= y_train_OH)

            class_weights_dic0 = dict(enumerate(class_weights))#calss weights for Out3
            #################################multiply in the 
            for key in class_weights_dic0:
                class_weights_dic0[key]=class_weights_dic0[key]*w_p[i]
            ##################################
            class_weights_dic.append(class_weights_dic0)#calss weights for Out3
    
        
        
        
        #######################remove test sample to reduce the computation cost of each epoch during calculation of the epoch testing accuracy
        if  val==0 and len(y_true[0])>30 : 
            rand_ind=np.random.permutation(len(y_true[0]))
            ind=rand_ind[0:30]
            def remove_item (ind, y_test):
                for i, item in enumerate(y_test):
                    y_test[i]= item[ind]
                return y_test
            y_test=remove_item (ind, y_test)
            y_true = remove_item (ind, y_true)
            x_test = remove_item( ind, x_test)
       ######################################################     
       ########base network only the last in:
        classifier.fit(x_train,y_train[2],x_test,y_test[2], y_true, class_weights_dic)
       
    
    def create_classifier(classifier_name, input_shape, nb_classes, nb_classes_last, output_directory, dataset_name, verbose = False):
        if classifier_name=='fcn': 
            from classifiers import fcn        
            return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='mlp':
            from  classifiers import  mlp 
            return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='resnet':
            from  classifiers import resnet 
            return resnet.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
        #############my resnet
        if classifier_name=='resnet_AT':
            from  classifiers import resnet_AT 
            return resnet_AT.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='resnet_AT2':
            from  classifiers import resnet_AT2 
            return resnet_AT2.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='resnet_AT4':#
            from  classifiers import resnet_AT4 
            return resnet_AT4.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)    
        if classifier_name=='resnet_AT0':# tre diffferent kernel
            from  classifiers import resnet_AT0 
            return resnet_AT0.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
       
        if classifier_name=='resnet_AT5':#test iblock original resnet
            from  classifiers import resnet_AT5 
            return resnet_AT5.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)      
    
        if classifier_name=='resnet_AT5_perm':#test iblock original resnet
            from  classifiers import resnet_AT5_perm 
            return resnet_AT5_perm.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose) 
    
        if classifier_name=='resnet_AT5_base_twoIn':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn 
            return resnet_AT5_base_twoIn.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)      
    
        if classifier_name=='resnet_AT5_base_twoIn2':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn2 
            return resnet_AT5_base_twoIn2.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)      
        
        if classifier_name=='resnet_AT5_base_twoIn3':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn3 
            return resnet_AT5_base_twoIn3.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn4':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn4 
            return resnet_AT5_base_twoIn4.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn6':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6 
            return resnet_AT5_base_twoIn6.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn6_2':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6_2 
            return resnet_AT5_base_twoIn6_2.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
     
        if classifier_name=='resnet_AT5_base_twoIn6_4':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6_4 
            return resnet_AT5_base_twoIn6_4.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
    
        if classifier_name=='resnet_AT5_base_twoIn6_5':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6_5 
            return resnet_AT5_base_twoIn6_5.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn6_8':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6_8 
            return resnet_AT5_base_twoIn6_8.Classifier_RESNET(output_directory,input_shape, nb_classes, dataset_name, root_dir, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn6_11':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6_11 
            return resnet_AT5_base_twoIn6_11.Classifier_RESNET(output_directory,input_shape, nb_classes, nb_classes_last, dataset_name, root_dir, verbose)     
                      
        if classifier_name=='resnet_AT5_base_twoIn6b2':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6b2
            return resnet_AT5_base_twoIn6b2.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        if classifier_name=='resnet_AT5_base_twoIn6b':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6b 
            return resnet_AT5_base_twoIn6b.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
    
        if classifier_name=='resnet_AT5_base_twoIn6c':#test iblock original resnet
            from  classifiers import resnet_AT5_base_twoIn6c 
            return resnet_AT5_base_twoIn6c.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
        
        if classifier_name=='resnet_AT2_con':#test iblock original resnet
            from  classifiers import resnet_AT2_con 
            return resnet_AT2_con.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT2_con_for':#test iblock original resnet
            from  classifiers import resnet_AT2_con_for 
            return resnet_AT2_con_for.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT2_add_for':#test iblock original resnet
            from  classifiers import resnet_AT2_add_for 
            return resnet_AT2_add_for.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT2_con_for_block':#test iblock original resnet
            from  classifiers import resnet_AT2_con_for_block
            return resnet_AT2_con_for_block.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT2_con_for_base':#test iblock original resnet
            from  classifiers import resnet_AT2_con_for_base
            return resnet_AT2_con_for_base.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT2_con_for_base2':#test iblock original resnet
            from  classifiers import resnet_AT2_con_for_base2
            return resnet_AT2_con_for_base2.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_AT5_base':#test iblock original resnet
            from  classifiers import resnet_AT5_base
            return resnet_AT5_base.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        
        if classifier_name=='resnet_AT2_con_for_layer':#test iblock original resnet
            from  classifiers import resnet_AT2_con_for_layer
            return resnet_AT2_con_for_layer.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)    
        if classifier_name=='resnet_perm':#test iblock original resnet
            from  classifiers import resnet_perm
            return resnet_perm.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)  
        if classifier_name=='resnet_permt':#test iblock original resnet
            from  classifiers import resnet_permt
            return resnet_permt.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)                   
        ########################################
        if classifier_name=='mcnn':
            from  classifiers import mcnn
            return mcnn.Classifier_MCNN(output_directory,verbose)
        if classifier_name=='tlenet':
            from  classifiers import tlenet
            return tlenet.Classifier_TLENET(output_directory,verbose)
        if classifier_name=='twiesn':
            from classifiers import twiesn
            return twiesn.Classifier_TWIESN(output_directory,verbose)
        if classifier_name=='encoder':
            from classifiers import encoder
            return encoder.Classifier_ENCODER(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='mcdcnn':
            from classifiers import mcdcnn
            return mcdcnn.Classifier_MCDCNN(output_directory,input_shape, nb_classes, verbose)
        if classifier_name=='cnn': # Time-CNN
            from classifiers import cnn
            return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)
    
    ############################################### main 

    #if 1==1:    
    # this is the code used to launch an experiment on a dataset
    
    itr = '_itr_8'

    if itr == '_itr_0': 
        itr = ''

    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+\
        dataset_name+'/'

    output_directory = create_directory(output_directory)

    print('Method: ',archive_name, dataset_name, classifier_name, itr)
    #if 1==1:
    import time
    start_time = time.time()
    if dataset_name == 'cornell':
        datasets_dict0 = read_dataset_cornell(root_dir,archive_name,dataset_name)
    else:

        datasets_dict0 = read_dataset(root_dir,archive_name,dataset_name)
    
 
##### # make the min to zero of labels        
    datasets_dict00={}
    
    y1,y2 = transform_labels(datasets_dict0[dataset_name][1],datasets_dict0[dataset_name][3])
    datasets_dict00[dataset_name] = (datasets_dict0[dataset_name][0], y1, datasets_dict0[dataset_name][2],y2)
    datasets_dict0 = datasets_dict00
    datasets_dict00={}
    
    #### make different permutation of training data and combine test data with the training data
    val=0# don't use the validation data
    print ('val=', val)

    down_sample = 1 # dpwnsample the trainig data to reduce the overal paire inputs
    datasets_dict, datasets_dict0 = new_combination_data_twoIn_twoOut2(val=val, down_sample=down_sample)# use different data0 to generate validation data which give an unseen data for validation #tow input network, if val=1, 20% of trained data is used as validation data
    ###############################
    fit_classifier(w_p)
    ##############################
    end_time = time.time()
    Elaps_time = end_time-start_time
    print('Elapsed time:', Elaps_time)
    print (classifier_name)
    print('DONE')

    # the creation of this directory means
    ######################################################
    create_directory(output_directory+'/DONE')
    ####################################################
    # test the trained model on the test input:

    #############calculate the output of the network for test data
    from keras.models import load_model
    from sklearn.metrics import accuracy_score

    model_dir = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+dataset_name+'/'+'best_model.hdf5'
    model = load_model(model_dir)
  ##########################test model on test data based on last output:
    if val==0:
       x_test=datasets_dict[dataset_name][2]
       y_test_t = datasets_dict[dataset_name][3]
    else:
       x_test=datasets_dict[dataset_name][4]
       y_test_t = datasets_dict[dataset_name][5]

    if len(x_test[0].shape) == 2: # if univariate 
       x_test[0]=np.expand_dims(x_test[0],axis=2)
       x_test[1]=np.expand_dims(x_test[1],axis=2)

    yo_test =  model.predict(x_test)
    y_actual = yo_test.argmax(axis=1)#base net

    
    ### calculate accuracy:
    if val ==0:
        y_test =  datasets_dict0[dataset_name][3] 
        if down_sample==1:
            y_resampled = datasets_dict[dataset_name][5]
    else:
        y_test =  datasets_dict0[dataset_name][5]
        if down_sample==1:
            y_resampled = datasets_dict[dataset_name][7]

        
    y_tru = y_test
            

   
    print ('calculating precision:')
    accuracy, precision_s, recall_s, f1_s = calculate_accuracy_precision(y_actual, y_tru, y_resampled)  
    #y_actual:actual binary label come from the final output(maine output) of the model
    #y_tru: is the clasificatin label for the samples
    #y_resampled: the clasificatin label for the representative samples for diferent classes
    
    
    ############ calculate based on the max probability achived:
    accuracy_onMax = calculate_accuracyMax3_single_o(y_actual, y_tru, y_resampled, yo_test)# single out/Max prob/last out
    #######save the output as scv file:
    data= { 'acc_last_vot':[accuracy],'pre_last_vot':[ precision_s],'recall_last_vot':[ recall_s], 'f1_last_vot':[ f1_s], 'acc_last_Max':[accuracy_onMax], 'Duration':[Elaps_time]}
  
    df = pd.DataFrame(data=data)
    df.to_csv(output_directory+'Ab_best_model.csv', index=False)
    ######################
    return accuracy, precision_s, recall_s, f1_s, accuracy_onMax, Elaps_time#o1, oLast, oLastMax, oLastMaxVot
