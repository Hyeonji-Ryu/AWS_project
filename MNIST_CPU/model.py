import tensorflow as tf
import pickle
import numpy as np

def get_data(data_name):
    
    with open(data_name, 'rb') as f:
        data = pickle.load(f)
    
    return data
	
def Model():
    
    Input = tf.keras.layers.Input(shape = (28,28,1))
    conv1 = tf.keras.layers.Conv2D(32, (3,3), kernel_initializer='he_normal',strides = 1, activation = 'relu')(Input)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3,3), kernel_initializer='he_normal',strides = 1, activation = 'relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, (3,3), kernel_initializer='he_normal',strides = 1, activation = 'relu')(pool2)
    flat = tf.keras.layers.Flatten()(conv3)
    den1= tf.keras.layers.Dense(64,kernel_initializer='he_normal', activation = 'relu')(flat)
    output = tf.keras.layers.Dense(10,kernel_initializer='he_normal', activation = 'softmax')(den1)
    
    model = tf.keras.Model(Input, output)
    
    return model

class Mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        
        if logs.get('val_accuracy') > 0.99 or logs.get('accuracy') > 0.9999 :
            print('done!')
            self.model.stop_training = True

# training

train_x = np.array(get_data('train_x.pickle')).astype(np.float32)
train_y = tf.one_hot(get_data('train_y.pickle'),10)
test_x = np.array(get_data('test_x.pickle')).astype(np.float32)
test_y = tf.one_hot(get_data('test_y.pickle'),10)

model  = Model()
callbacks = Mycallbacks()
model.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_x, train_y, epochs = 100, batch_size = 100, validation_data=(test_x, test_y), callbacks= [callbacks])

tf.saved_model.save(model, 'mnist_convNet')
