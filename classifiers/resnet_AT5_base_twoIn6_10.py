# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import pandas as pd 
import time

import matplotlib 
#matplotlib.use('agg')
import matplotlib.pyplot as plt 

#from utils.utils import save_logs
#from utils_AT import save_logs_AT_o1o3
import sys
sys.path.append('../')#append the parent directory to the path then you could import from utils_AT whic is on the paret directory
  
# importing
from utils_AT import save_logs_AT_o1o3
###############
from keras.models import load_model
from keras import regularizers

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


class Classifier_RESNET: 

   def __init__(self, output_directory, input_shape, nb_classes, nb_classes_last,  dataset_name, root_dir, verbose=False):
      self.output_directory = output_directory
      self.model = self.build_model(input_shape, nb_classes, nb_classes_last, dataset_name, root_dir)
      if(verbose==True):
         self.model.summary()
      self.verbose = verbose
      self.model.save_weights(self.output_directory+'model_init.hdf5')

   def build_model(self, input_shape, nb_classes, nb_classes_last, dataset_name, root_dir):
      n_feature_maps = 64
      
#      ########################################load model:
#      load_model_dir = root_dir+'/results/1500resnet_AT5/UCR_TS_Archive_2015_itr_8/'+dataset_name+'/best_model.hdf5'
#      #######load for input1:
#      # 1 block:
#      model1=load_model(load_model_dir)
#      ###resnet with higher number of layers+1500 epoches:
#      for layer in model1.layers:
##          layer.trainable = False
#          layer.name = layer.name+'a'#two have different name from the second branch
#      model1.layers[-1].name = 'dense_1a'
#      ######remove the lat two layers of loaded model:
##      model1.layers.pop()
##      model1.layers.pop()
##      print(model1.output)
##      model1.layers[-1].outbound_nodes = []
##      model1.layers[-1].trainable= True   
      ###############################
      # BLOCK 1 
      input_layer1 = keras.layers.Input(input_shape)
      conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer1)
#      conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=10, padding='same')(input_layer1)

      
      conv_x = BatchNormalization()(conv_x)
      conv_x = keras.layers.Activation('relu')(conv_x)
      
      conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
      conv_y = BatchNormalization()(conv_y)
      conv_y = keras.layers.Activation('relu')(conv_y)
      
      conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
      conv_z = BatchNormalization()(conv_z)
      
      # expand channels for the sum 
      shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer1)
      shortcut_y = BatchNormalization()(shortcut_y)
      
      output_block_1 = keras.layers.add([shortcut_y, conv_z])
      output_block_1 = keras.layers.Activation('relu')(output_block_1)
      # FINAL 
      
      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)
      output_layer_1 = keras.layers.Dense(nb_classes, activation='softmax', name='dense_1a')(gap_layer)


      ############# load for input2:
#      model2=load_model(load_model_dir)
#      for layer in model2.layers:
#          layer.trainable = False
#          layer.name = layer.name+'b'#two have different name from the second branch
#      model2.layers[-1].name = 'dense_2b'
#      ######remove the lat two layers of loaded model:
#      model2.layers.pop()
#      model2.layers.pop()
##      print(model2.output)
#      model2.layers[-1].outbound_nodes = []      
      
      ##############################################
      
      # BLOCK 1 
      input_layer2 = keras.layers.Input(input_shape)
      
      conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer2)
#      conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=10, padding='same')(input_layer2)
      
      
      conv_x = BatchNormalization()(conv_x)
      conv_x = keras.layers.Activation('relu')(conv_x)
      
      conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
      conv_y = BatchNormalization()(conv_y)
      conv_y = keras.layers.Activation('relu')(conv_y)
      
      conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
      conv_z = BatchNormalization()(conv_z)
      
      # expand channels for the sum 
      shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer2)
      shortcut_y = BatchNormalization()(shortcut_y)
      
      output_block_2 = keras.layers.add([shortcut_y, conv_z])
      output_block_2 = keras.layers.Activation('relu')(output_block_2)
      
      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_2)
      output_layer_2 = keras.layers.Dense(nb_classes, activation='softmax', name='dense_2b')(gap_layer)
        
      '''     
      
      # BLOCK 1 

      conv_x = keras.layers.Conv1D(filters=int(n_feature_maps/2), kernel_size=8, padding='same')(input_layer1)
      conv_x = keras.layers.normalization.BatchNormalization()(co
      conv_x2 = keras.layers.Activation('relu')(conv_x2)
      conv_x = keras.layers.concatenate([conv_x, conv_x2])      nv_x)
      conv_x = keras.layers.Activation('relu')(conv_x)
      
      conv_x2 = keras.layers.Conv1D(filters=int(n_feature_maps/2), kernel_size=8, padding='same')(input_layer2)
      conv_x2 = keras.layers.normalization.BatchNormalization()(conv_x2)
      
      '''
      
#      conv_x_fix = keras.layers.concatenate([model1.layers[-1].output, model2.layers[-1].output]) 
#      conv_x_fix = keras.layers.concatenate([model1.layers[-3].output, model2.layers[-3].output]) 
#      conv_x_fix = keras.layers.concatenate([model1.layers[-3].output, model2.layers[-1].output]) 
#      conv_x_fix =model2.layers[-1].output

#      conv_x_fix = keras.layers.subtract([model1.layers[-3].output, model2.layers[-1].output]) 
      conv_x_fix = keras.layers.subtract([output_block_1, output_block_2]) 
      
      
#      conv_x_fix = keras.layers.Dropout(0.5)(conv_x_fix)
      '''
      conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
      conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
      conv_y = keras.layers.Activation('relu')(conv_y)

      conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
      conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

      # expand channels for the sum 
      shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer1)
      shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

      output_block_1 = keras.layers.add([shortcut_y, conv_z])
      output_block_1 = keras.layers.Activation('relu')(output_block_1)

      # BLOCK 2 

#      conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
#      conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
#      conv_x = keras.layers.Activation('relu')(conv_x)
#
#      conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
#      conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
#      conv_y = keras.layers.Activation('relu')(conv_y)
#
#      conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
#      conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
#
#      # expand channels for the sum 
#      shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
#      shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
#
#      output_block_2 = keras.layers.add([shortcut_y, conv_z])
#      output_block_2 = keras.layers.Activation('relu')(output_block_2)
#
#      # BLOCK 3 
      '''
#      conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same', kernel_regularizer=regularizers.l2(0.01))(conv_x_fix)
      conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(conv_x_fix)
#      conv_x = keras.layers.Dropout(0.2)(conv_x)
      conv_x = BatchNormalization()(conv_x)
      conv_x = keras.layers.Activation('relu')(conv_x)

      conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
#      conv_y = keras.layers.Dropout(0.2)(conv_y)
      conv_y = BatchNormalization()(conv_y)
      conv_y = keras.layers.Activation('relu')(conv_y)

#      conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
      conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)

#      conv_z = keras.layers.Dropout(0.2)(conv_z)
      conv_z = BatchNormalization()(conv_z)

      # no need to expand channels because they are equal 
      shortcut_y = BatchNormalization()(conv_x_fix)

      output_block_3 = keras.layers.add([shortcut_y, conv_z])
      output_block_3 = keras.layers.Activation('relu')(output_block_3)
      
      
      # FINAL 
      
#      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)
      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

      output_layer = keras.layers.Dense(nb_classes_last, activation='softmax', name='dense_last')(gap_layer)

#      model = keras.models.Model(inputs=[model1.input, model2.input], outputs=output_layer)
#      model = keras.models.Model(inputs=[model1.input, model2.input], outputs=[model1.layers[-1].output, model2.layers[-1].output, output_layer])
#      model = keras.models.Model(inputs=[model1.input, model2.input], outputs=[model1.layers[-1].output, output_layer])
#      model = keras.models.Model(inputs=[input_layer1,input_layer2], outputs=[output_layer_1, output_layer])
      model = keras.models.Model(inputs=[input_layer1,input_layer2], outputs=[output_layer_1, output_layer_2, output_layer])


      model.compile(loss='categorical_crossentropy', optimizer=Adam(), 
         metrics=['accuracy'])

      reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
#      reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.01)

      file_path = self.output_directory+'best_model.hdf5' # if you get error in this line delete the folder that exist: C:/Users/ERD204/Documents/data/time_series/results/resnet_AT5_base_twoIn6_10/UCR_TS_Archive_2015_itr_8/ArrowHead/
#
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
      
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='dense_last_loss', save_best_only=True)
      
      ###########out1
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='dense_1a_loss', save_best_only=True)#out1
      
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_dense_1a_loss', save_best_only=True)#out1
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_dense_1a_acc', save_best_only=True)
      
     #####out3
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_dense_last_loss', save_best_only=True)
#      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_dense_last_acc', save_best_only=True)
      

      self.callbacks = [reduce_lr,model_checkpoint]
#      model.summary()


      return model
  
#   def fit(self, x_train, y_train, x_val, y_val,y_true): class_weights
   def fit(self, x_train, y_train, x_val, y_val,y_true, class_weights):
      # x_val and y_val are only used to monitor the test loss and NOT for training  
#      batch_size = 16
#      batch_size = 30
      batch_size = 100

#      nb_epochs = 1500
      nb_epochs = 300
      

      #nb_epochs = 100
      
      #nb_epochs = 300
      
#      nb_epochs = 150
#      nb_epochs = 400
      
      #nb_epochs = 50
      
#      nb_epochs = 10
#      nb_epochs = 15
      
      print ('Number of Epoches = ', nb_epochs)        

#      mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
      mini_batch_size = int(min(x_train[0].shape[0]/10, batch_size))

      start_time = time.time() 

      # hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
      #    verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, class_weight=class_weights, shuffle=True)

      hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
         verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, shuffle=True)


      duration = time.time() - start_time

      model = keras.models.load_model(self.output_directory+'best_model.hdf5')

      y_pred = model.predict(x_val)

      # convert the predicted from binary to integer 
#      y_pred = np.argmax(y_pred , axis=1)
#      y_pred[2] = np.argmax(y_pred[2] , axis=1)#only calculat for the main output
      y_pred[1] = np.argmax(y_pred[1] , axis=1)#only calculat for the main output o1o3
          

#      save_logs(self.output_directory, hist, y_pred, y_true, duration)
#      save_logs_AT(self.output_directory, hist, y_pred[2], y_true[2], duration)#only calculat for the main output
      save_logs_AT_o1o3(self.output_directory, hist, y_pred[1], y_true[1], duration)#only calculat for the main output
      

      keras.backend.clear_session()

