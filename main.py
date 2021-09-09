import os
import cv2
import random
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K

import spatial_attention_unet


tf.test.is_gpu_available()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
val_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory("DUTS-TR/train_frames", batch_size = 32)

train_mask_generator = train_datagen.flow_from_directory('DUTS-TR/train_masks', batch_size = 32)



val_image_generator = val_datagen.flow_from_directory('DUTS-TR/val_frames', batch_size = 32)

val_mask_generator = val_datagen.flow_from_directory('DUTS-TR/val_masks', batch_size = 32)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
  
    while (True):
        img = np.zeros((batch_size, 512, 512, 3)).astype('float')
        mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

            train_img = cv2.imread(img_folder+'/'+n[i])/255.
            train_img =  cv2.resize(train_img, (512, 512))# Read an image from folder and resize

            img[i-c] = train_img #add to array - img[0], img[1], and so on.


            train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            train_mask = cv2.resize(train_mask, (512, 512))
            train_mask = train_mask.reshape(512, 512, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

            mask[i-c] = train_mask

        c+=batch_size
        if(c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n)
                  # print "randomizing again"
        yield img, mask
        
train_frame_path = 'DUTS-TR/train_frames/train_frames'
train_mask_path = 'DUTS-TR/train_masks/train_masks'

val_frame_path = 'DUTS-TR/val_frames/val_frames'
val_mask_path = 'DUTS-TR/val_masks/val_masks'

# Train the model
train_gen = data_gen(train_frame_path,train_mask_path, batch_size = 4)
val_gen = data_gen(val_frame_path,val_mask_path, batch_size = 4)
        
NO_OF_TRAINING_IMAGES = len(os.listdir('DUTS-TR/train_frames/train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir('DUTS-TR/val_frames/val_frames/'))

print(NO_OF_TRAINING_IMAGES)
print(NO_OF_VAL_IMAGES)

NO_OF_EPOCHS = 50
BATCH_SIZE = 4

weights_path = '/DUTS-TR'

m = spatial_attention_unet.SA_UNet(input_size = (512,512,3))
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
  
m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'mae', f1_metric])

checkpoint = ModelCheckpoint(weights_path, monitor='METRIC_TO_MONITOR', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
     
