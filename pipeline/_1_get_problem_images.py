import glb
import os
import random
from processing.problem_to_image import ProblemToImage
import uuid
from tqdm import tqdm
from joblib import Parallel, delayed

glb.delete_files_in_dir(glb.problem_images_dir)
glb.delete_files_in_dir(glb.problem_data_dir)

n = 50000

def apply(_):
    problem = [random.randint(1, glb.max_problem_num) for _ in range(4)]
    problem_to_image  = ProblemToImage(problem)
    problem_to_image.set_img()
    base_filename = str(uuid.uuid4()) 
    problem_to_image.save(os.path.join(glb.problem_images_dir, base_filename + '.png'))           
    glb.save_json(problem, os.path.join(glb.problem_data_dir, base_filename + '.json'))
    
apply(0)
    
Parallel(
    n_jobs=8, 
    backend='loky')(delayed(apply)(_) for _ in  tqdm(range(n)))

 