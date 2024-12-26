import glb
import os
from tensorflow.keras.models import Model
import tensorflow as tf
from processing.train_test_data import TrainTestData
import numpy as np

cases = glb.open_json(os.path.join(glb.data_dir, 'cases.json'))

latents = []
train_test_data = TrainTestData(cases)
train_test_data.set_from_cases()
X, _  = train_test_data.get()

for i in range(0, glb.n_models_for_bagging):
    model = tf.keras.models.load_model(os.path.join(glb.models_dir, 'latent_{}.keras'.format(i)), compile=False)
    latent_model = Model(inputs=model.input, outputs=model.get_layer('latent').output)
    model_latents = latent_model.predict(X)
    latents.append(model_latents)
    
latents = np.mean(latents, axis=0)

for i, _case in enumerate(cases):
    _case['latent'] = latents[i].tolist()


glb.save_json(cases, os.path.join(glb.data_dir, 'cases.json'))
