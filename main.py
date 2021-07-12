import tensorflow as tf
from tensorflow import keras
import cv2
from scipy import io
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
from preprocess import get_data
import gc

'''
Dataset can downloaded from the following repo :
!git clone https://github.com/bearpaw/clothing-co-parsing.git
'''

# Read the labels from the matlab label file and save it as a dictionary
mat = io.loadmat('./clothing-co-parsing/label_list.mat')
labels = {0:'background'}
for i in range(1, len(mat['label_list'][0])):
  labels[i] = mat['label_list'][0][i][0]

#Read the images and append them to a list
images = []
for i in range(1,1001):
        url = './clothing-co-parsing/photos/%04d.jpg'%(i)
        img = cv2.imread(url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(tf.convert_to_tensor(img))

#Read the masks and append them to a list
masks = []
for i in range(1,1001):
        url = './clothing-co-parsing/annotations/pixel-level/%04d.mat'%(i)
        file = io.loadmat(url)
        mask = tf.convert_to_tensor(file['groundtruth'])
        masks.append(mask)

# Perform preprocessing and data augmentation and create training and validation data
train, val = get_data(images, masks)

#Free up memory
del images, masks
gc.collect()

#Prepare data batches and shuffle the training data
BATCH = 32
BUFFER = 1000
STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH
train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
val = val.batch(BATCH)


#Create an auto-encoder model for segmentation

def VGG16(x):
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)

        return x, a, b

def decoder(x, a, b):
        
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)
        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)
        
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(y)
        
        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)

        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
            
        x = Conv2D(59, (3, 3), padding='same')(x)

        return x

#Initialize the encoder model and load pre-trained weights
net_input = Input(shape=(256, 256, 3))
vgg_output = VGG16(net_input)
model = Model(inputs=net_input, outputs=vgg_output, name='model')

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')

model.load_weights(vgg_weights_path, by_name=True)

#Train only the higher layers of the encoder to adjust to the dataset
unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']

for layer in model.layers:
        if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
x, a, b = model.output

x = decoder(x, a, b)

#Join the encoder and decoder networks to form the complete network and start model training
vision_model = Model(inputs=net_input, outputs=x, name='vision_model')

opt = RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.)

vision_model.compile(loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             optimizer=opt,
             metrics=['accuracy'])

# I use early stopping to tackle overfitting
early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=12, verbose=0, mode='auto')

#Decrease learning rate by a factor from 0.1 when the loss starts oscillating
redu = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

vision_model.fit(train, validation_data=val,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEPS, callbacks=[early, redu],
                epochs=50)

vision_model.save('./task2_model.h5')