import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from enum import Enum
import json
import shutil

data_dir = 'gitignore'
case_images_dir = os.path.join(data_dir, 'case_images')
problem_images_dir = os.path.join(data_dir, 'problem_images')
problem_data_dir =  os.path.join(data_dir, 'problem_data')
models_dir = os.path.join(data_dir, 'models')
results_dir = 'results'
cloud_dir = r'C:\Users\ofire\Google Drive\LLM Paper\24-puzzle-bu'

class LabelMethods(Enum):
    any_cat = 0
    any_cat_and_decomp = 1
     
class LLMProviders(Enum):
    groq = 0
    together_ai = 1
    
test_case_seeds = [262431, 299542, 850113, 781658, 955662, 620838, 457165, 424223, 114535, 158928, 949459, 487385, 227188, 989997, 155924, 222123, 998847, 475629, 356080, 767944, 480458, 386314, 764604, 142658, 341209, 997829, 323471, 348989, 546259, 95100, 170713, 557911, 450825, 785528, 949953, 780024, 180692, 836521, 595480, 999221, 14607, 721022, 331986, 917427, 572911, 66299, 120359, 311129, 543947, 145655]

n_test_cases = 30
max_pixel_value=255
img_dim = 90
n_models_for_bagging = 1
max_problem_num = 13
nums_per_problem = 4

count_llm_queries = 0

operators = ['*', '+', '-', '/']   

classes = [
    (24, 0, 1),
    (24, 0, 2),
    (24, 0, 3),
    (24, 1, 2),
    (24, 1, 3),
    (24, 2, 3),
    
    (12, 0, 1),
    (12, 0, 2),
    (12, 0, 3),
    (12, 1, 2),
    (12, 1, 3),
    (12, 2, 3),
    
    (8, 0, 1),
    (8, 0, 2),
    (8, 0, 3),
    (8, 1, 2),
    (8, 1, 3),
    (8, 2, 3),
    
    (6, 0, 1),
    (6, 0, 2),
    (6, 0, 3),
    (6, 1, 2),
    (6, 1, 3),
    (6, 2, 3),
    
    ]


def delete_files_in_dir(_dir):
    filenames = os.listdir(_dir)
    if not os.path.exists(_dir):
        return
    for filename in filenames:
        path = os.path.join(_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
            
def copy_files_from_dir(src_dir, dest_dir):
    for item in os.listdir(src_dir):
        src_file = os.path.join(src_dir, item)
        if os.path.isfile(src_file):
            dest_file = os.path.join(dest_dir, item)
            try:
                shutil.copy(src_file, dest_file)
            except Exception as e:
                raise e               
            
def set_gpu(set_tf_gpu_allocator=True, set_virtual_device_configuration=True):
    if set_tf_gpu_allocator:
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)==0:
        print('GPU NOT SET!!!')
    for gpu in gpus:
        try:
            if not tf.config.experimental.get_memory_growth(gpu):
                tf.config.experimental.set_memory_growth(gpu, True)
                if set_virtual_device_configuration:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)]
                    )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)         
            
def img_to_arr(img):
    img = img.resize((img_dim, img_dim))
    x = image.img_to_array(img)
    x = x / max_pixel_value  
    x = np.expand_dims(x, axis=0)          
    return x   

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def save_json(d, path):
    data = json.dumps(d, indent=4)
    with open(path, 'w') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())        