import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
Dataset can downloaded from the following repo :
!git clone https://github.com/bearpaw/clothing-co-parsing.git
'''

#Helper functions for preprocessing and data augmentation
def resize_image(image):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    image = tf.image.resize(image, (256,256))
    return image 
     
def resize_mask(mask):
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, (256,256))
    mask = tf.cast(mask, tf.uint8)
    return mask

def brightness(img, mask):
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask
 
def gamma(img, mask):
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask

def hue(img, mask):
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask

def crop(img, mask):
    img = tf.image.central_crop(img, 0.7)
    img = tf.image.resize(img, (256,256))
    mask = tf.image.central_crop(mask, 0.7)
    mask = tf.image.resize(mask, (256,256))
    mask = tf.cast(mask, tf.uint8)
    return img, mask

def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

def flip_vert(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask


#Main function to call helper functions and generate data slices of training and validation data
def get_data(images, masks):
    X = [resize_image(i) for i in images]
    y = [resize_mask(m) for m in masks]

    train_X, val_X,train_y, val_y = train_test_split(X,y,test_size=0.2,random_state=0)
    train_X = tf.data.Dataset.from_tensor_slices(train_X)
    val_X = tf.data.Dataset.from_tensor_slices(val_X)
    train_y = tf.data.Dataset.from_tensor_slices(train_y)
    val_y = tf.data.Dataset.from_tensor_slices(val_y)

    train = tf.data.Dataset.zip((train_X, train_y))
    val = tf.data.Dataset.zip((val_X, val_y))

    # perform augmentation on train data only
    a = train.map(brightness)
    b = train.map(gamma)
    c = train.map(hue)
    d = train.map(crop)
    e = train.map(flip_hori)
    f = train.map(flip_vert)
    g = train.map(rotate)

    train = train.concatenate(a)
    train = train.concatenate(b)
    train = train.concatenate(c)
    train = train.concatenate(d)
    train = train.concatenate(e)
    train = train.concatenate(f)
    train = train.concatenate(g)

    return train, val