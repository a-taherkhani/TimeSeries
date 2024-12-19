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

from utils.utils import save_logs
#from utils_AT import save_logs_AT_o1o3

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
      
#      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)
#      output_layer_1 = keras.layers.Dense(nb_classes, activation='softmax', name='dense_1a')(gap_layer)

      
      ##############################################
      
      # BLOCK 1 
      input_layer2 = keras.layers.Input(input_shape)
      
      conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer2)
      
      
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
      
      conv_x_fix = keras.layers.subtract([output_block_1, output_block_2]) 
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
      conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(conv_x_fix)
      conv_x = BatchNormalization()(conv_x)
      conv_x = keras.layers.Activation('relu')(conv_x)

      conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
      conv_y =BatchNormalization()(conv_y)
      conv_y = keras.layers.Activation('relu')(conv_y)

      conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)

      conv_z =BatchNormalization()(conv_z)

      # no need to expand channels because they are equal 
      shortcut_y =BatchNormalization()(conv_x_fix)

      output_block_3 = keras.layers.add([shortcut_y, conv_z])
      output_block_3 = keras.layers.Activation('relu')(output_block_3)
      
      
      # FINAL 
      
#      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1)
      gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

      output_layer = keras.layers.Dense(nb_classes_last, activation='softmax', name='dense_last')(gap_layer)

      model = keras.models.Model(inputs=[input_layer1,input_layer2], outputs=output_layer)


      model.compile(loss='categorical_crossentropy', optimizer=Adam(), 
         metrics=['accuracy'])

      reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
      print('self.output_directory: ',self.output_directory)

      file_path = self.output_directory+'best_model.hdf5' 
#
      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)#base model
      

      self.callbacks = [reduce_lr,model_checkpoint]


      return model
  
   def fit(self, x_train, y_train, x_val, y_val,y_true, class_weights):
      # x_val and y_val are only used to monitor the test loss and NOT for training  
      batch_size = 100
      nb_epochs = 300
      
      print ('Number of Epoches = ', nb_epochs)        

      mini_batch_size = int(min(x_train[0].shape[0]/10, batch_size))

      start_time = time.time() 


      hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
         verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks, shuffle=True)
          
      duration = time.time() - start_time

      model = keras.models.load_model(self.output_directory+'best_model.hdf5')

      y_pred = model.predict(x_val)

      # convert the predicted from binary to integer 
      y_pred = np.argmax(y_pred , axis=1)#base net
          
      
      save_logs(self.output_directory, hist, y_pred, y_true[2], duration)#base net
      

      keras.backend.clear_session()

