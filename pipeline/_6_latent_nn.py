import glb
import os
from processing.custom_metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from processing.train_test_data import TrainTestData
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import resample
import random

cases = glb.open_json(os.path.join(glb.data_dir, 'cases.json'))

train_cases = [_case for _case in cases if _case['is_test']==False]
test_cases = [_case for _case in cases if _case['is_test']==True]

train_data = TrainTestData(train_cases)
train_data.set_from_cases()
train_X, train_Y = train_data.get()

test_data = TrainTestData(test_cases)
test_data.set_from_cases()
test_X, test_Y = test_data.get()

n_input = train_X.shape[1]
n_output = train_Y.shape[1]

epochs=500
dropouts = [0.15, 0.05]
lambdas = [0.01, 0.001]
batch_sizes = [64, 32]

for i in range(0, glb.n_models_for_bagging):

    sample_X, sample_Y = resample(train_X, train_Y)
    
    inputs = layers.Input(shape=n_input, name='inputs')
    a = inputs
    a = layers.Dense(64, activation='relu', name='latent', kernel_regularizer=regularizers.L2(random.choice(lambdas)))(a)
    a = layers.LayerNormalization()(a)
    a = layers.Dropout(random.choice(dropouts))(a)
    outputs = layers.Dense(n_output, activation='sigmoid', name='output')(a)
        
    model = Model(inputs=inputs, outputs=outputs)

    learning_rate = 0.01
    
    optimizer = Adam(learning_rate=learning_rate)
    loss = BinaryCrossentropy()
        
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=10, min_lr=1e-4, verbose=1)
        
    model.compile(optimizer=optimizer,
                  loss=loss, 
                  metrics=[BinaryPrecision(),
                           BinaryRecall(),
                           BinaryF1Score()])
    
    model.fit(sample_X,
              sample_Y,
              epochs=epochs, 
              batch_size=random.choice(batch_sizes),
              validation_data=(test_X, test_Y),   
              callbacks=[reduce_lr],
             )
    
    model.save(os.path.join(glb.models_dir, 'latent_{}.keras'.format(i)))