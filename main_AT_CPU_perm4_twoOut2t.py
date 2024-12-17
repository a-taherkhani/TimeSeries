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

import numpy as np
import sys
import sklearn 

#####utils_AT
from utils_AT import gen_val_set2, gen_val_set, plot_confusion_matrix, data_permutations, balance_data, test_data_periparation,test_data_periparation_twoOut,  calculate_accuracy#, test_balance_data

#############randome seed:
#seed = 100
seed = 50
np.random.seed(seed)
#TensorFlow has its own random number generator
from tensorflow import set_random_seed
set_random_seed(seed)
####################
def data_preparation(datasets_dict, dataset_name):#mix original x_tes with train data 
#        from utils.utils import transform_labels

        x_train = datasets_dict[dataset_name][0]
        y_train = datasets_dict[dataset_name][1]
        x_test = datasets_dict[dataset_name][2]
        y_test = datasets_dict[dataset_name][3]
        ######    ######AT data permutation
#        x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)
#        x_test = x_testt
#        y_test = y_testt
#    
#        x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)
#        jjjj #
#        x_train = x_traint
#        y_train = y_traint
        
    
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
#    a=datasets_dict1[dataset_name][0]



#    datasets_dict00 = {'Adiac': (x_train, y_train, x_test, y_test)};
#     my_dict = {	dataset_name: (x_train, y_train, x_test, y_test)};
#     a=my_dict['Adiac'][0]
    
    
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
#    index = np.random.permutation(int(len(y_test)))
#    
#    y_testt=y_testt[index]
#    
#    x_test1=x_test1[index,:]
#    
#    x_test2=x_test2[index,:]
#    
#    x_testt = x_testt[index,:]
###################
#    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt, balance=6)
    x_test = [x_test1, x_test2]
############################################
    y_test = y_testt

#    x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)#
    x_train1, x_train2, x_traint, y_traint = test_data_periparation(x_train, y_train, x_train, y_train)#it considered (i,i)

#    x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint, balance=6)

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
#    index = np.random.permutation(int(len(y_test)))
#    
#    y_testt=y_testt[index]
#    
#    x_test1=x_test1[index,:]
#    
#    x_test2=x_test2[index,:]
#    
#    x_testt = x_testt[index,:]
###################
#    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt, balance=6)
    x_test = [x_test1, x_test2]
############################################
#    y_test = y_testt
    y_test =[ y_test1, y_test2, y_testt ]

#    x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)#
#    x_train1, x_train2, x_traint, y_traint = test_data_periparation(x_train, y_train, x_train, y_train)#it considered (i,i)
    x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)

#    x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint, balance=6)

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

def new_combination_data_twoIn_twoOut2(val=0):# make an input list that contine x1 and x2
    #1)combine test data with trainig data
    #2)generat different permutation of train data and make it imbalance
    if val == 0:
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        ######    ######AT data permutation
    #    x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
        
        ##################################
    #    index = np.random.permutation(int(len(y_test)))
    #    
    #    y_testt=y_testt[index]
    #    
    #    x_test1=x_test1[index,:]
    #    
    #    x_test2=x_test2[index,:]
    #    
    #    x_testt = x_testt[index,:]
    ###################
    #    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt, balance=6)
        x_test = [x_test1, x_test2]
    ############################################
    #    y_test = y_testt
        y_test =[ y_test1, y_test2, y_testt ]
    
    #    x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)#
    #    x_train1, x_train2, x_traint, y_traint = test_data_periparation(x_train, y_train, x_train, y_train)#it considered (i,i)
        x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)
    
    #    x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint, balance=6)
    
        ######################
        x_train = [x_train1, x_train2]
        ###########################
        y_train = y_traint
        y_train =[ y_train1, y_train2, y_traint]
        ###############################
#        if val==1:
#            print ('randomlly select validation set' )
#            x_train, y_train, x_val, y_val = gen_val_set(x_train, y_train)
#            datasets_dict1 = {	dataset_name: (x_train, y_train, x_val, y_val, x_test, y_test)};
#        else:
            ############################
        datasets_dict1 = {	dataset_name: (x_train, y_train, x_test, y_test)};
    
        
        return datasets_dict1, datasets_dict0
    elif val==1:
        x_train = datasets_dict0[dataset_name][0]
        y_train = datasets_dict0[dataset_name][1]
        x_test = datasets_dict0[dataset_name][2]
        y_test = datasets_dict0[dataset_name][3]
        ################        
        print ('randomlly select validation set' )
        x_train, y_train, x_val, y_val = gen_val_set2(x_train, y_train)
        datasets_dict0_val = {dataset_name:((x_train, y_train, x_val, y_val, x_test, y_test))}
#        datasets_dict1 = {	dataset_name: (x_train, y_train, x_val, y_val, x_test, y_test)};

        ######    ######AT data permutation
    #    x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
        x_test1, x_test2, x_testt, y_test1, y_test2, y_testt = test_data_periparation_twoOut(x_test, y_test, x_train, y_train)#
        
        ##################################
    #    index = np.random.permutation(int(len(y_test)))
    #    
    #    y_testt=y_testt[index]
    #    
    #    x_test1=x_test1[index,:]
    #    
    #    x_test2=x_test2[index,:]
    #    
    #    x_testt = x_testt[index,:]
    ###################
    #    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt, balance=6)
        x_test = [x_test1, x_test2]
        y_test =[ y_test1, y_test2, y_testt ]
  ###########validation data:
        x_val1, x_val2, x_valt, y_val1, y_val2, y_valt = test_data_periparation_twoOut(x_val, y_val, x_train, y_train)#
        x_val = [x_val1, x_val2]
        y_val =[ y_val1, y_val2, y_valt ]
    ############################################   
    #    x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)#
    #    x_train1, x_train2, x_traint, y_traint = test_data_periparation(x_train, y_train, x_train, y_train)#it considered (i,i)
        x_train1, x_train2, x_traint, y_train1, y_train2, y_traint = test_data_periparation_twoOut(x_train, y_train, x_train, y_train)#it considered (i,i)
    
    #    x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint, balance=6)
    
        ######################
        x_train = [x_train1, x_train2]
        ###########################
        y_train = y_traint
        y_train =[ y_train1, y_train2, y_traint]
        ###############################
#        if val==1:
#            print ('randomlly select validation set' )
#            x_train, y_train, x_val, y_val = gen_val_set(x_train, y_train)
#            datasets_dict1 = {	dataset_name: (x_train, y_train, x_val, y_val, x_test, y_test)};
#        else:
            ############################
#        datasets_dict1 = {	dataset_name: (x_train, y_train, x_test, y_test)};
        datasets_dict1 = {	dataset_name: (x_train, y_train, x_val, y_val, x_test, y_test)};
    
        
        return datasets_dict1, datasets_dict0_val


#def total_test_data():# make an input list that contine [x1, x2]. 390*391
#    #1)combine test data with trainig data
#    #2)generat different permutation of train data and make it imbalance
#    x_train = datasets_dict0[dataset_name][0]
#    y_train = datasets_dict0[dataset_name][1]
#    x_test = datasets_dict0[dataset_name][2]
#    y_test = datasets_dict0[dataset_name][3]
#    ######    ######AT data permutation
#    x_test1, x_test2, x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
#    ##################################
#    x_test = [x_test1, x_test2]
#############################################
#    x_test00 = []
#    for i, modality in enumerate(x_test):
#            x_test00.append(np.expand_dims(modality, axis=2))
#    return x_test00
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
     #    x_test1, x_test2, x_testt, y_testt = balance_data(x_test1, x_test2, x_testt, y_testt)
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
    
def fit_classifier(): 
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
#    ######    ######AT data permutation
#    x_testt, y_testt = test_data_periparation(x_test, y_test, x_train, y_train)#
#    x_test = x_testt
#    y_test = y_testt
#
#    x_train1, x_train2, x_traint, y_traint = data_permutations(x_train, y_train)
#    x_train1, x_train2, x_traint, y_traint=balance_data(x_train1, x_train2, x_traint, y_traint)
#    x_train = x_traint
#    y_train = y_traint
#    
#
#    ####################################
#    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))
    nb_classes = len(np.unique(np.concatenate((y_train[2],y_test[2]),axis =0)))

    # make the min to zero of labels
#    y_train,y_test = transform_labels(y_train,y_test)
    for i, (yTrain, yTest) in enumerate( zip(y_train, y_test)):
        y_train[i],y_test[i] = transform_labels(yTrain, yTest)

    # save orignal y because later we will use binary
#    y_true = y_test.astype(np.int64) 
    y_true =[]
    for i, yTrue in enumerate( y_test):
        y_true.append(yTrue.astype(np.int64) )
#        y_true[i]=yTrue.astype(np.int64) 
    # transform the labels from integers to one hot vectors
#    enc = sklearn.preprocessing.OneHotEncoder()
#    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
#    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
#    y_test = enc.transform(y_test.reshape(-1,1)).toarray()
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
#    input_shape = x_train.shape[1:]
    input_shape = x_train[0].shape[1:]
            
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)
    
    #########generate Class_weights:
    from sklearn.utils import class_weight
    class_weights_dic =[]
    for i, yTrain in enumerate( y_train):

        y_train_OH =np.argmax(yTrain, axis=1)# One hot encoded vector
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train_OH),
                                                 y_train_OH)
        class_weights_dic0 = dict(enumerate(class_weights))#calss weights for Out3
        class_weights_dic.append(class_weights_dic0)#calss weights for Out3

    
    
    
#    from sklearn.utils import class_weight
#    y_train_OH =np.argmax(y_train[2], axis=1)# One hot encoded vector
#    class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_train_OH),
#                                                 y_train_OH)
#    class_weights_dic = dict(enumerate(class_weights))
        
#    Thirdly and lastly add it to the model fitting
#    model.fit(X_train, y_train, class_weight=class_weights)
#    classifier.fit(x_train,y_train,x_test,y_test, y_true)
    classifier.fit(x_train,y_train,x_test,y_test, y_true, class_weights_dic)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False):
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

    if classifier_name=='resnet_AT5_base_twoIn6_4t':#test iblock original resnet
        from  classifiers import resnet_AT5_base_twoIn6_4t 
        return resnet_AT5_base_twoIn6_4t.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)     
       
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

# change this directory for your machine
# it should contain the archive folder containing both univariate and multivariate archives
#root_dir = '/mnt/nfs/casimir/'
#root_dir = 'C:/Users/ERD204/Documents/data/time_series'# labtop
root_dir = 'C:/Users/cmp3tahera/Documents/data/time_series'#cpu

"""
if sys.argv[1]=='transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1]=='visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1]=='viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1]=='viz_cam':
    viz_cam(root_dir)
elif sys.argv[1]=='generate_results_csv':
    res = generate_results_csv('results.csv',root_dir)
    print(res)
else:
    """
    #python3 main.py UCR_TS_Archive_2015 Coffee fcn _itr_8
if 1==1:    
    # this is the code used to launch an experiment on a dataset
#    archive_name = sys.argv[1]
#    dataset_name = sys.argv[2]
#    classifier_name=sys.argv[3]
#    itr = sys.argv[4]
    archive_name = 'UCR_TS_Archive_2015'#'mts_archive'#'UCR_TS_Archive_2015'
    #dataset_name = 'ArabicDigits'
    dataset_name ='ArrowHead'#'Adiac'#'wafer'##'50words'#'ArrowHead'#'Coffee'
#    dataset_name ='Adiac'
#    dataset_name ='Beef'
#    dataset_name ='Car'#'Adiac'#'wafer'##'50words'#'ArrowHead'#'Coffee'
#    dataset_name ='ECG200'#'CBF'#'Adiac'#'wafer'##'50words'#'ArrowHead'#'Coffee'

#    classifier_name='resnet'#'fcn'
#    classifier_name='resnet_AT5_base_twoIn4bt'#'resnet_AT5_base_twoIn3'#'resnet_AT5_base_twoIn2'#'resnet_AT5_base_twoIn'#'resnet_permt'#'resnet_AT5_perm'
    #'resnet_AT2_con_for'#'resnet_AT2_con_for_layer'#'resnet_AT2_con_for'#'resnet_AT2_con_for_base'#'resnet_AT2_con_for'#'resnet_AT5_base'#'resnet_AT2_con_for_base'#'resnet_AT2_con_for'#'resnet_AT2'#'resnet_AT2_con_for_block'#'resnet_AT2_add_for'#'resnet_AT2_con_for'#resnet_AT2_con'#'resnet_AT0'#'resnet_AT2'#'fcn'
#    classifier_name='resnet_AT5_base_twoIn6b'#'resnet_AT5_base_twoIn3'#'resnet_AT5_base_twoIn2'#'resnet_AT5_base_twoIn'#'resnet_permt'#'resnet_AT5_perm'
#    classifier_name='resnet_AT5_base_twoIn6b2'#   out 1 make good increase 89.7% in 10 epoch 
#    classifier_name='resnet_AT5_base_twoIn6'#   
#    classifier_name='resnet_AT5_base_twoIn6_2'#   
    classifier_name='resnet_AT5_base_twoIn6_4t'#   

    
    itr = '_itr_8'

    if itr == '_itr_0': 
        itr = ''

    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+\
        dataset_name+'/'

    output_directory = create_directory(output_directory)

    print('Method: ',archive_name, dataset_name, classifier_name, itr)

#    if output_directory is None: 
#        print('Already done')
    if 1==1:
#    else: 
        import time
        start_time = time.time()
        datasets_dict0 = read_dataset(root_dir,archive_name,dataset_name)
 
    ##### # make the min to zero of labels        
        datasets_dict00={}
        
        y1,y2 = transform_labels(datasets_dict0[dataset_name][1],datasets_dict0[dataset_name][3])
        datasets_dict00[dataset_name] = (datasets_dict0[dataset_name][0], y1, datasets_dict0[dataset_name][2],y2)
        datasets_dict0 = datasets_dict00
        datasets_dict00={}
        
        #### make different permutation of training data and combine test data with the training data
#        datasets_dict=new_combination_data()
#        datasets_dict = new_combination_data_twoIn()#tow input network
        val=1# use the validation data
#        val=0
        
#        datasets_dict = new_combination_data_twoIn_twoOut(val=val)#tow input network, if val=1, 20% of trained data is used as validation data
#        datasets_dict = new_combination_data_twoIn_twoOut2(val=val)# use different data0 to generate validation data which give an unseen data for validation #tow input network, if val=1, 20% of trained data is used as validation data
        datasets_dict, datasets_dict0 = new_combination_data_twoIn_twoOut2(val=val)# use different data0 to generate validation data which give an unseen data for validation #tow input network, if val=1, 20% of trained data is used as validation data
        
        ###############################
        fit_classifier()
        end_time = time.time()
#        print ('Elapsed time is:')
#        print (end_time-start_time)
        print('Elapsed time:', end_time-start_time)
        print (classifier_name)
        print('DONE')

        # the creation of this directory means
        create_directory(output_directory+'/DONE')
        # test the trained model on the test input:
        
        ####combine the test data with train data to test the network
               
#        x_test=total_test_data()
#        x_test, y_test=total_test_data2()

        #############calculate the output of the network for test data
        from keras.models import load_model
        from sklearn.metrics import accuracy_score
#        model = load_model('C:/Users/ERD204/Documents/data/time_series/results/resnet_AT5_perm/UCR_TS_Archive_2015_itr_8/Adiac/best_model.hdf5')

        model_dir = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+dataset_name+'/'+'best_model.hdf5'
        model = load_model(model_dir)
#        model.summary()
        '''
        #############training data accuracy:
        x_train = datasets_dict[dataset_name][0]
        yo_train =  model.predict(x_train)
        yo=np.argmax(yo_train, axis = 1)
        yTrain = datasets_dict[dataset_name][1]
        acc = accuracy_score(yo, yTrain)
        ###############
        '''
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
        y_actual = yo_test[2].argmax(axis=1)
        
        ### calculate accuracy:
        if val ==0:
            y_test =  datasets_dict0[dataset_name][3] 
        else:
            y_test =  datasets_dict0[dataset_name][5]
            
        y_tru = y_test
        y_train =  datasets_dict0[dataset_name][1] 
           # make the min to zero of labels
#        y_train,y_tru = transform_labels(y_train,y_tru)

        
        print('Testing accuracy of the last output:')
        accuracy = calculate_accuracy(y_actual, y_tru, y_train)#actual and true label:(y_actual, y_tru)
        
        ####calculate testing accuracy on out1:
        y_actual1 = yo_test[0].argmax(axis=1)
        y_true_o1= y_test_t[0]
        acc1 = accuracy_score(y_actual1, y_true_o1)
        print('\nTesting accuracy of Out1: ')
        print (acc1)

        '''
        ##########accuracy on train data:################################################################
        x_test=total_test_data(train=1)
        
        yo_test =  model.predict(x_test)
        y_actual = yo_test[2].argmax(axis=1)
        
        ### calculate accuracy:
        y_test =  datasets_dict0[dataset_name][1] #lable of trained data
        
        # make the min to zero of labels
        y_train,y_tru = transform_labels(y_train,y_tru)
       
        y_tru = y_test
        accuracy = calculate_accuracy(y_actual, y_tru, y_train)#actual and true label:(y_actual, y_tru)
        '''
        ######confusion matrix:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Compute confusion matrix
#        x_test, y_test=total_test_data2()
        y_true_o3= y_test_t[2]#true out put lable for o3
        cnf_matrix = confusion_matrix(y_true_o3, y_actual)
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        class_names =[0,1]
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')  
        plt.show()