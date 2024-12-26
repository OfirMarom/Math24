import glb
import os
from PIL import Image
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random

glb.set_gpu(set_tf_gpu_allocator=False, set_virtual_device_configuration=True)

epochs=10
batch_size=32

def generator(image_filenames, shuffle=True):
    while True:
        if shuffle:
            random.shuffle(image_filenames)
        batch_start = 0
        batch_end = batch_size
        while batch_start < len(image_filenames):
            batch_image_filenames = image_filenames[batch_start:batch_end] 
            yield get_X_Y(batch_image_filenames)
            batch_start += batch_size
            batch_end += batch_size
            
def get_X_Y(image_filenames): 
    X = []
    Y = []
    for image_filename in image_filenames:
        img_path = os.path.join(glb.problem_images_dir, image_filename)
        base_filename, _ = os.path.splitext(image_filename)
        data_filename = base_filename + '.json'
        data_path = os.path.join(glb.problem_data_dir, data_filename)
        data = glb.open_json(data_path)
        img = Image.open(img_path)
        x = glb.img_to_arr(img)
        y = []
        for num in data:
            oh = [0] * glb.max_problem_num
            oh[num-1]=1
            y.append(oh)    
        X.append(x)
        Y.append(y)
        
    X = np.vstack(X)
    
    Y = np.array(Y) 
    Y = [Y[:, i, :] for i in range(Y.shape[1])]
    
    return X, Y

image_filenames = os.listdir(glb.problem_images_dir)

train_image_filenames, val_image_filenames = train_test_split(image_filenames, 
                                                              test_size=0.2, 
                                                              random_state=523)

train_generator = generator(train_image_filenames, shuffle=True)
val_generator = generator(val_image_filenames, shuffle=False)

steps_per_epoch_train = len(train_image_filenames) // batch_size
steps_per_epoch_val = len(val_image_filenames) // batch_size

outputs = []
inputs = layers.Input(shape=(glb.img_dim, glb.img_dim, 1))

x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs) 
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x) 
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x) 
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
for i in range(glb.nums_per_problem):    
    output = layers.Dense(glb.max_problem_num, activation='softmax', name='output{}'.format(i+1))(x)
    outputs.append(output)

model = Model(inputs=inputs, outputs=outputs) 

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

 
model.fit(train_generator,
          steps_per_epoch=steps_per_epoch_train,
          validation_data=val_generator,
          validation_steps=steps_per_epoch_val,epochs=epochs,
          )

model.save(os.path.join(glb.models_dir, 'cnn.keras'))