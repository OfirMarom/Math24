import glb
import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np

cnn = tf.keras.models.load_model(os.path.join(glb.models_dir, 'cnn.keras'), compile=False)

with open(os.path.join(glb.data_dir, 'cases.json'), 'r') as file:
    cases = json.load(file)

X = []
for _case in cases:
    img_filename = _case['problem_img_filename']
    img_path = os.path.join(glb.case_images_dir, img_filename)
    img = Image.open(img_path)
    img = img.resize((glb.img_dim, glb.img_dim))
    x = glb.img_to_arr(img)
    X.append(x)

X = np.vstack(X)
    
preds = cnn.predict(X)

success = 0
for i, _case in enumerate(cases):
    pred1 = preds[0][i]
    pred2 = preds[1][i]
    pred3 = preds[2][i]
    pred4 = preds[3][i]
    nums = np.argmax([pred1,pred2,pred3,pred4],axis=1) + 1
    nums = nums.tolist()
    nums = sorted(nums)
    _case['problem_from_img'] = nums
    if nums == _case['problem']:
        success+=1
        
print('success rate: {}'.format(success/len(cases)))       
        
data = json.dumps(cases, indent=4)
with open(os.path.join(glb.data_dir, 'cases.json'), 'w') as f:
    f.write(data)     
    f.flush()
    os.fsync(f.fileno())     