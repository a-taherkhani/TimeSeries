# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:46:13 2018

@author: ERD204
"""

from  itertools import permutations, combinations, groupby
import numpy as np
from utils.utils import calculate_metrics
from utils.utils import plot_epochs_metric
import matplotlib.pyplot as plt      
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# permute the training data: 390*389 new data is put in xt, yt 
def read_dataset_cornell(root_dir,archive_name,dataset_name):
    datasets_dict = {}

  
    if dataset_name == 'cornell':
        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
         ##############data:
        data = loadmat(file_name + 'data.mat')
        data = data['data']
        
        labelTV = loadmat(file_name +'labelTV.mat')
        labelTV = labelTV['labelTV']
        labelTV = np.squeeze(labelTV)
        #labelTV = [labelTV]
        #labelTV=np.squeeze(labelTV)
        num_classes = labelTV.max()
        labelTV = labelTV-1
        x_train, x_test, y_train, y_test = train_test_split(data, labelTV, test_size=0.3, random_state=0)
#        X_train = np.expand_dims(X_train, 3)
#        X_test =np.expand_dims(X_test, 3)
#        # Convert class vectors to binary class matrices.
#        y_train = keras.utils.to_categorical(y_train, num_classes)
#        y_test = keras.utils.to_categorical(y_test, num_classes)
        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
            y_test.copy())

#    else:
#        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'+dataset_name
#        x_train, y_train = readucr(file_name+'_TRAIN')
#        x_test, y_test = readucr(file_name+'_TEST')
#        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
#            y_test.copy())
        
   
    return datasets_dict

def data_permutations(x, y):
    perm = permutations([i for i in range(x.shape[0])], 2)
    
    x1=[]
    x2=[]
    xt = []
    yt=[]
    dim = np.shape(x)[1]
#    dim = np.shape(x)[1:]
    
    # put different permutation of input data x in the xt
    for i in perm:
        xt.append(np.reshape(x[i,:],(1,2*dim)))
        x1.append(x[i,:][0])
        x2.append(x[i,:][1])
#        print (y[list(i)][0])
#        if (y[list(i)][0] == y[list(i)][1]):
#            print (y[list(i)][0] == y[list(i)][1])
            
        yt.append(y[list(i)][0] == y[list(i)][1])
        
#        yt.append()
        
#    yt = np.multiply(yt,1)+1
    yt = np.multiply(yt,1)
        
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    xt = np.asarray(xt)
    xt = np.squeeze(xt)
    return x1, x2, xt, yt

# make balanced. Make an equall number of yes/no class (balanced dat)
def balance_data(x1,x2,xt,yt, balance=1):
    
    index_yes=list(np.where(yt==1))
    index_yes =  np.asanyarray(index_yes)
    index_yes=index_yes.squeeze(axis=0)
    
    index_no = list(np.where(yt==0))
    rand_index=np.random.permutation(index_no[0])
    index_no = rand_index[:len(index_yes)*balance]
     
    index =np.append(index_yes, index_no)
    index = np.random.permutation(index)
    
    yt=yt[index]
    
    x1=x1[index,:]
    
    x2=x2[index,:]
    
    xt = xt[index,:]
    return x1, x2, xt, yt

# make balanced. Make an equall number of yes/no class (balanced dat)
#def test_balance_data(xt,yt):
#    
#    index_yes=list(np.where(yt==1))
#    index_yes =  np.asanyarray(index_yes)
#    index_yes=index_yes.squeeze(axis=0)
#    
#    index_no = list(np.where(yt==0))
#    rand_index=np.random.permutation(index_no[0])
#    index_no = rand_index[:len(index_yes)]
#     
#    index =np.append(index_yes, index_no)
#    index = np.random.permutation(index)
#    
#    yt=yt[index]
#    
##    x1=x1[index,:]
##    
##    x2=x2[index,:]
#    
#    xt = xt[index,:]
#    return xt, yt

# prepare data for testing. for each testing data all the training data is used: 
#[repeated x_test[i]][x_train]
def test_data_periparation(x, y, x_train, y_train):
#    perm = permutations([i for i in range(x.shape[0])], 2)
   xt = [] 
   yt = []
   x1=[]
   x2=[]
   
   y1_test =[]
   y2_train = []
   
   for data, l_test in zip(x,y):
       data = np.expand_dims(data, axis = 0)
#       data = np.reshape(data, (len(data),1))
       x_test00 = np.repeat(data, x_train.shape[0], axis=0)
       x_test0 = np.concatenate((x_test00, x_train), axis =1)#[repeated x_test(i),x_train]
       
       y_test0= np.repeat(l_test, x_train.shape[0],axis =0)
       if len(xt)==0:
           xt.append(x_test0)
           xt = np.asarray(xt)
           xt = np.squeeze(xt)
#################################
           x1.append(x_test00)
           x1 = np.asarray(x1)
           x1 = np.squeeze(x1)

           x2.append(x_train)
           x2 = np.asarray(x2)
           x2 = np.squeeze(x2)
################################           
           #labl:
           y1_test.append(y_test0)
           y1_test = np.asarray(y1_test)
           y1_test = np.squeeze(y1_test)
           
           y2_train = y_train
       else:
           xt = np.concatenate((xt,x_test0), axis = 0)
##############################
           x1 = np.concatenate((x1,x_test00), axis = 0)
           
           x2 = np.concatenate((x2,x_train), axis = 0)
##########################           
           
           y1_test = np.concatenate((y1_test,y_test0))#
           y2_train = np.concatenate((y2_train,y_train))# repetition of train label
   yt = y1_test == y2_train
#   yt = np.multiply(yt,1)+1# convert the boolian to number started from 1
   yt = np.multiply(yt,1)# convert the boolian to number started from 1

##    x1 = np.asarray(x1)
##    x2 = np.asarray(x2)
#   xt = np.asarray(xt)
#    xt = np.squeeze(xt)
   return x1, x2, xt, yt
# preipare data for testing. for each testing data all the training data is used: 
#[repeated x_test[i]][x_train], [reapeat y_est[i]], y_train
def test_data_periparation_twoOut_machine(x, y, x_train, y_train):
#    perm = permutations([i for i in range(x.shape[0])], 2)
   xt = [] 
   yt = []
   x1=[]
   x2=[]

   
   y1_test =[]
   y2_train = []
   
   for data, l_test in zip(x,y):
       data = np.expand_dims(data, axis = 0)
#       data = np.reshape(data, (len(data),1))
       x_test00 = np.repeat(data, x_train.shape[0], axis=0)
       x_test0 = np.concatenate((x_test00, x_train), axis =1)#[repeated x_test(i),x_train]
       
       y_test0= np.repeat(l_test, x_train.shape[0],axis =0)
       if len(xt)==0:
           xt.append(x_test0)
           xt = np.asarray(xt)
           xt = np.squeeze(xt)
#################################
           x1.append(x_test00)
           x1 = np.asarray(x1)
           x1 = np.squeeze(x1)
           
#           y1.append(y_test00)

           x2.append(x_train)
           x2 = np.asarray(x2)
           x2 = np.squeeze(x2)
            
#           y2.append(y_train)
################################           
           #labl:
           y1_test.append(y_test0)
           y1_test = np.asarray(y1_test)
           y1_test = np.squeeze(y1_test)
           
           y2_train = y_train
       else:
           xt = np.concatenate((xt,x_test0), axis = 0)
##############################
           x1 = np.concatenate((x1,x_test00), axis = 0)
           
           x2 = np.concatenate((x2,x_train), axis = 0)
##########################           
           
           y1_test = np.concatenate((y1_test,y_test0))#
           y2_train = np.concatenate((y2_train,y_train))# repetition of train label
   yt = y1_test - y2_train#for regrision/analog output
   #yt = np.multiply(yt,1)# convert the boolian to number started from 1


   return x1, x2, xt, y1_test, y2_train, yt
def test_data_periparation_twoOut(x, y, x_train, y_train):
#    perm = permutations([i for i in range(x.shape[0])], 2)
   xt = [] 
   yt = []
   x1=[]
   x2=[]

   
   y1_test =[]
   y2_train = []
   
   for data, l_test in zip(x,y):
       data = np.expand_dims(data, axis = 0)
#       data = np.reshape(data, (len(data),1))
       x_test00 = np.repeat(data, x_train.shape[0], axis=0)
       x_test0 = np.concatenate((x_test00, x_train), axis =1)#[repeated x_test(i),x_train]
       
       y_test0= np.repeat(l_test, x_train.shape[0],axis =0)
       if len(xt)==0:
           xt.append(x_test0)
           xt = np.asarray(xt)
           xt = np.squeeze(xt)
#################################
           x1.append(x_test00)
           x1 = np.asarray(x1)
           x1 = np.squeeze(x1)
           
#           y1.append(y_test00)

           x2.append(x_train)
           x2 = np.asarray(x2)
           x2 = np.squeeze(x2)
            
#           y2.append(y_train)
################################           
           #labl:
           y1_test.append(y_test0)
           y1_test = np.asarray(y1_test)
           y1_test = np.squeeze(y1_test)
           
           y2_train = y_train
       else:
           xt = np.concatenate((xt,x_test0), axis = 0)
##############################
           x1 = np.concatenate((x1,x_test00), axis = 0)
           
           x2 = np.concatenate((x2,x_train), axis = 0)
##########################           
           
           y1_test = np.concatenate((y1_test,y_test0))#
           y2_train = np.concatenate((y2_train,y_train))# repetition of train label
   yt = y1_test == y2_train
#   yt = np.multiply(yt,1)+1# convert the boolian to number started from 1
   yt = np.multiply(yt,1)# convert the boolian to number started from 1

##    x1 = np.asarray(x1)
##    x2 = np.asarray(x2)
#   xt = np.asarray(xt)
#    xt = np.squeeze(xt)
#x1: repeated test, y1_test: repeated lable for test
#x2: repeated train, y2_train: reapeated label for train

   return x1, x2, xt, y1_test, y2_train, yt

def gen_val_set(x_train, y_train):#x_train = [x_train1, x_train2], y_train =[ y_train1, y_train2, y_traint]
    #20% of training data is used for validation
    indix= np.random.permutation(len(y_train[0]))
    indix_val = indix[0:int(0.20*len(y_train[0]))]#30% of traaining data as validation data
    y_val = []
    for i, itm_y in enumerate(y_train):
            y_val.append( itm_y[indix_val])
            y_train[i]=np.delete(itm_y, indix_val)
    
    x_val =[]
    for i, item in enumerate(x_train):
        x_val.append(item[indix_val]) 
        x_train[i]=np.delete(item, indix_val,axis=0)
    return x_train, y_train, x_val, y_val
def gen_val_set2(x_train, y_train):#x_train = [x_train1, x_train2], y_train =[ y_train1, y_train2, y_traint]
    #20% of training data is used for validation
    indix= np.random.permutation(len(y_train))
    indix_val = indix[0:int(0.20*len(y_train))]#30% of traaining data as validation data
#    y_val = []
#    y_val.append( y_train[indix_val])
    y_val =y_train[indix_val]
    y_train=np.delete(y_train, indix_val)
    
 
    
#    x_val =[]
#    x_val.append(x_train[indix_val])
    x_val=x_train[indix_val]
    x_train=np.delete(x_train, indix_val,axis=0)

    return x_train, y_train, x_val, y_val

def calculate_accuracy(y_actual, y_tru, y_train):#actual and true label:(y_actual, y_tru)
        #input 
        #  1) actual ouput of the comparing network, [0,1],
        #  2) tru out put of the test samples, 3) training input data lable
        # output: the accracy
        
            from itertools import groupby
            count = 0
            y_actual = y_actual.reshape((len(y_tru), -1))#391*390

            for j, yOut in enumerate( y_actual):#for each testing sample

                indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]
                y_train_1.sort()
                if len(y_train_1)!=0:
    
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    if key[ind] == y_tru[j]:
                        count = count+1
                        
                    
#                    print(y_train_1)#non zero output 
#                    print (rep)#number of repeatation
#                    print (key)
#                    
#                    print (key[ind])#key of maximume repitation
#                    print (y_tru[j])
##                   
#                    print('************')
 
            accuracy = count/len(y_tru)#*100
            print (accuracy)
            return accuracy
def calculate_accuracy_precision(y_actual, y_tru, y_train):#actual and true label:(y_actual, y_tru)
        #input
        
            #y_actual:actual binary label come from the final output(maine output) of the model#1) actual ouput of the comparing network, [0,1],
            #y_tru: is the clasificatin label for the samples. 2) tru out put of the test samples,
            #y_train(y_resampled): the clasificatin label for the representative samples for diferent classes        #   3) training input data lable

        #  
        # output: the accracy
        
            from itertools import groupby
            count = 0
            y_actual = y_actual.reshape((len(y_tru), -1))#391*390
            y_pred_new=[]
            y_actual_new=[]

            for j, yOut in enumerate( y_actual):#for each testing sample

                indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]
                y_train_1.sort()
                if len(y_train_1)!=0:
    
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    ######
                    y_pred_new.append(key[ind])#append the predected lable
                    y_actual_new.append(y_tru[j])
                    if key[ind] == y_tru[j]:
                        count = count+1
                        
                    
#                    print(y_train_1)#non zero output 
#                    print (rep)#number of repeatation
#                    print (key)
#                    
#                    print (key[ind])#key of maximume repitation
#                    print (y_tru[j])
##                   
#                    print('************')
####calculate precision, recall and f1 on the data samples which are predected by the final layer
            precision_s=precision_score(y_actual_new, y_pred_new, average='macro')
            recall_s=recall_score(y_actual_new, y_pred_new, average='macro')
            f1_s=f1_score(y_actual_new, y_pred_new, average='macro')
 
            accuracy = count/len(y_tru)#*100
            #print (accuracy)
            return accuracy, precision_s, recall_s, f1_s
        
#        print (j)
def calculate_accuracy_multi(y_actual, y_tru, y_train):#actual and true label:(y_actual, y_tru)
        #input 
        #  1) actual ouput of the comparing network: n_class^2
        #  2) tru out put of the test samples, 3) training input data lable
        # output: the accracy

            y_last=n_class*y_test1+y_test2
            

        
            from itertools import groupby
            count = 0
            y_actual = y_actual.reshape((len(y_tru), -1))#391*390

            for j, yOut in enumerate( y_actual):#for each testing sample

                indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]
                y_train_1.sort()
                if len(y_train_1)!=0:
    
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    if key[ind] == y_tru[j]:
                        count = count+1
 
            accuracy = count/len(y_tru)#*100
            print (accuracy)
            return accuracy
def calculate_accuracy_y(y_actual, y_tru, y_train):#actual and true label:(y_actual, y_tru)
        #input 
        #  1) actual ouput of the comparing network, [0,1],
        #  2) tru out put of the test samples, 3) training input data lable
        # output: the accracy
            y_predected=np.zeros(y_tru.shape)
            count = 0
            y_actual = y_actual.reshape((len(y_tru), -1))#391*390

            for j, yOut in enumerate( y_actual):#for each testing sample

                indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]#predected label
                y_train_1.sort()
                if len(y_train_1)!=0:
    
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    y_predected[j]=key[ind]#predected label with maximume repetition
                    if key[ind] == y_tru[j]:
                        count = count+1
                        
                    
#                    print(y_train_1)#non zero output 
#                    print (rep)#number of repeatation
#                    print (key)
#                    
#                    print (key[ind])#key of maximume repitation
#                    print (y_tru[j])
##                   
#                    print('************')
 
            accuracy = count/len(y_tru)#*100
            print (accuracy)
            return accuracy, y_predected
        ##########################################
def save_logs_AT(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    import pandas as pd
    from utils.utils import calculate_metrics
    from utils.utils import plot_epochs_metric

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

#    index_best_model = hist_df['loss'].idxmin() 
#    index_best_model = hist_df['val_loss'].idxmin() 
    
    index_best_model = hist_df['val_dense_1a_loss'].idxmin() 
#    index_best_model = hist_df['val_dense_last_acc'].idxmax() 
    
    
    
    
    row_best_model = hist_df.loc[index_best_model]
#    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0], 
#        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
#        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    
    

    df_best_model = pd.DataFrame(data = np.zeros((1,14),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss1', 'best_model_val_loss1', 'best_model_train_acc1', 
        'best_model_val_acc1', 'best_model_train_loss2', 'best_model_val_loss2', 'best_model_train_acc2', 
        'best_model_val_acc2', 'best_model_train_loss3', 'best_model_val_loss3', 'best_model_train_acc3', 
        'best_model_val_acc3', 'best_model_learning_rate','best_model_nb_epoch'])
    
#    df_best_model['best_model_train_loss'] = row_best_model['loss']
#    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
#    df_best_model['best_model_train_acc'] = row_best_model['acc']
#    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_train_loss1'] = row_best_model['dense_1a_loss']
    df_best_model['best_model_val_loss1'] = row_best_model['val_dense_1a_loss']
    df_best_model['best_model_train_acc1'] = row_best_model['dense_1a_acc']
    
    df_best_model['best_model_val_acc1'] = row_best_model['val_dense_1a_acc']
    ######
    index_best_model2 = hist_df['dense_1a_loss'].idxmin()
    row_best_model2 = hist_df.loc[index_best_model2]
#    print('\nvalidation accuracy of out1 based on dens_1a_loss:')
#    print (row_best_model2['val_dense_1a_acc'])
#    
#    
#    print ('\nvalidation accuracy of out1(val_dense_1a_acc):')
#    print (row_best_model['val_dense_1a_acc'])
    
   
    

    #################
    df_best_model['best_model_train_loss2'] = row_best_model['dense_2b_loss']
    df_best_model['best_model_val_loss2'] = row_best_model['val_dense_2b_loss']
    df_best_model['best_model_train_acc2'] = row_best_model['dense_2b_acc']
    df_best_model['best_model_val_acc2'] = row_best_model['val_dense_2b_acc']    

    df_best_model['best_model_train_loss3'] = row_best_model['dense_last_loss']
    df_best_model['best_model_val_loss3'] = row_best_model['val_dense_last_loss']
    df_best_model['best_model_train_acc3'] = row_best_model['dense_last_acc']
    df_best_model['best_model_val_acc3'] = row_best_model['val_dense_last_acc']    
    
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory+'epochs_loss.png')

    return df_metrics

def save_logs_AT_o1o3(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    import pandas as pd
    from utils.utils import calculate_metrics
    from utils.utils import plot_epochs_metric

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['dense_last_loss'].idxmin() 
    
   
    row_best_model = hist_df.loc[index_best_model]

    
    df_best_model = pd.DataFrame(data = np.zeros((1,10),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss1', 'best_model_val_loss1', 'best_model_train_acc1', 
        'best_model_val_acc1',  
        'best_model_train_loss3', 'best_model_val_loss3', 'best_model_train_acc3', 
        'best_model_val_acc3', 'best_model_learning_rate','best_model_nb_epoch'])
  
    df_best_model['best_model_train_loss1'] = row_best_model['dense_1a_loss']
    df_best_model['best_model_val_loss1'] = row_best_model['val_dense_1a_loss']
    # df_best_model['best_model_train_acc1'] = row_best_model['dense_1a_acc']
    df_best_model['best_model_train_acc1'] = row_best_model['dense_1a_accuracy']
    # df_best_model['best_model_val_acc1'] = row_best_model['val_dense_1a_acc']
    df_best_model['best_model_val_acc1'] = row_best_model['val_dense_1a_accuracy']
    
    ######


    df_best_model['best_model_train_loss3'] = row_best_model['dense_last_loss']
    df_best_model['best_model_val_loss3'] = row_best_model['val_dense_last_loss']
    # df_best_model['best_model_train_acc3'] = row_best_model['dense_last_acc']
    # df_best_model['best_model_val_acc3'] = row_best_model['val_dense_last_acc']    
    df_best_model['best_model_train_acc3'] = row_best_model['dense_last_accuracy']
    df_best_model['best_model_val_acc3'] = row_best_model['val_dense_last_accuracy']    
    
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory+'epochs_loss.png')

    return df_metrics
    #################################### 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def calculate_accuracyMax(y_actual, y_tru, y_train, yo_test):#calculate the accuray of the last out(2 classes out put), MAx prob
            y_actual_pro = yo_test[1]# probability for the actual output
#            y_actual_pro = yo_test[2]# probability for the actual output/ three out
    
            count = 0
            Num_test = len(y_tru)
            
            for i in range(Num_test):#number of test data
                y_pro=y_actual_pro[ i*len(y_train):(i+1)*len(y_train)]
                ind_max=np.argmax(y_pro[:,1])#if two input are from the sam class the out put is 1 exept 0, 
                actual_lable = y_train[ind_max]
                if actual_lable == y_tru[i]:
                      count = count+1
            accuracy_onMax = count/len(y_tru)#*100
            #print('The accuracy of the last out put based on the maximum probability is:')
            #print (accuracy_onMax)
            return accuracy_onMax 
        
def calculate_accuracyMaxVot(y_actual, y_tru, y_train, yo_test):#calculate the accuray of the last out(2 classes out put), MAx prob+voting
            y_actual_pro = yo_test[1]# probability for the actual output
#            y_actual_pro = yo_test[2]# probability for the actual output/3 out
    
            count = 0
            Num_test = len(y_tru)
            from itertools import groupby
            
            for i in range(Num_test):#number of test data
                y_pro=y_actual_pro[ i*len(y_train):(i+1)*len(y_train)]
                ind_max=np.argmax(y_pro[:,1])#if two input are from the sam class the out put is 1 exept 0, 
                #####
                th= np.maximum(0.9*y_pro[ind_max,1],0.5)
                indix=np.where(y_pro[:,1]>=th)#ind_max_v
            #    actual_lable = y_train[ind_max]
                
            #    indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]
                y_train_1.sort()
                if len(y_train_1)!=0:
            
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    if key[ind] == y_tru[i]:
                        count = count+1
            #    
            #    if actual_lable == y_tru[i]:
            #          count = count+1
            accuracy_onMax_v = count/len(y_tru)#*100
            #print('The accuracy of the last output based on the maximum probability and voting is:')
            #print( accuracy_onMax_v) 
            return accuracy_onMax_v 
    
def calculate_accuracyMax3(y_actual, y_tru, y_train, yo_test):#calculate the accuray of the last out(2 classes out put), MAx prob
#            y_actual_pro = yo_test[1]# probability for the actual output
            y_actual_pro = yo_test[2]# probability for the actual output/ three out
    
            count = 0
            Num_test = len(y_tru)
            
            for i in range(Num_test):#number of test data
                y_pro=y_actual_pro[ i*len(y_train):(i+1)*len(y_train)]
                ind_max=np.argmax(y_pro[:,1])#if two input are from the sam class the out put is 1 exept 0, 
                actual_lable = y_train[ind_max]
                if actual_lable == y_tru[i]:
                      count = count+1
            accuracy_onMax = count/len(y_tru)#*100
            #print('The accuracy of the last out put based on the maximum probability is:')
            #print (accuracy_onMax)
            return accuracy_onMax 
def calculate_accuracyMax3_single_o(y_actual, y_tru, y_train, yo_test):#calculate the accuray of the last out(2 classes out put), MAx prob/single output
#            y_actual_pro = yo_test[1]# probability for the actual output
#            y_actual_pro = yo_test[2]# probability for the actual output/ three out
            y_actual_pro = yo_test# probability for the actual output/ single out (o3)
    
            count = 0
            Num_test = len(y_tru)
            
            for i in range(Num_test):#number of test data
                y_pro=y_actual_pro[ i*len(y_train):(i+1)*len(y_train)]
                ind_max=np.argmax(y_pro[:,1])#if two input are from the sam class the out put is 1 exept 0, 
                actual_lable = y_train[ind_max]
                if actual_lable == y_tru[i]:
                      count = count+1
            accuracy_onMax = count/len(y_tru)#*100
            #print('The accuracy of the last out put based on the maximum probability is:')
            #print (accuracy_onMax)
            return accuracy_onMax         
        
def calculate_accuracyMaxVot3(y_actual, y_tru, y_train, yo_test):#calculate the accuray of the last out(2 classes out put), MAx prob+voting
#            y_actual_pro = yo_test[1]# probability for the actual output
            y_actual_pro = yo_test[2]# probability for the actual output/3 out
    
            count = 0
            Num_test = len(y_tru)
            from itertools import groupby
            
            for i in range(Num_test):#number of test data
                y_pro=y_actual_pro[ i*len(y_train):(i+1)*len(y_train)]
                ind_max=np.argmax(y_pro[:,1])#if two input are from the sam class the out put is 1 exept 0, 
                #####
                th= np.maximum(0.9*y_pro[ind_max,1],0.5)
                indix=np.where(y_pro[:,1]>=th)#ind_max_v
            #    actual_lable = y_train[ind_max]
                
            #    indix = np.where(yOut==1)#the indix of the trained data that has the same class of the applied test sample
                y_train_1=y_train[indix]
                y_train_1.sort()
                if len(y_train_1)!=0:
            
                    rep=[len(list(group)) for key, group in groupby(y_train_1)]
                    key = [key for key, group in groupby(y_train_1)]
                    
                    ind= np.argmax(rep)
                    if key[ind] == y_tru[i]:
                        count = count+1
            #    
            #    if actual_lable == y_tru[i]:
            #          count = count+1
            accuracy_onMax_v = count/len(y_tru)#*100
            print('The accuracy of the last output based on the maximum probability and voting is:')
            print( accuracy_onMax_v) 
            return accuracy_onMax_v 

           
