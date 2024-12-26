import glb
import subprocess

def run_pipeline(test_case_seed, models_dir=None):
    
    glb.delete_files_in_dir(glb.models_dir)
    if models_dir is not None:
        glb.copy_files_from_dir(models_dir, glb.models_dir)    
    
    if models_dir is None:
        subprocess.run(['python', r'pipeline\_1_get_problem_images.py'], check=True)
        subprocess.run(['python', r'pipeline\_2_cnn.py'], check=True)
    
    subprocess.run(['python', r'pipeline\_3_build_cases.py', str(test_case_seed)], check=True)
    subprocess.run(['python', r'pipeline\_4_cnn_pred.py'], check=True)
    subprocess.run(['python', r'pipeline\_5_set_features.py'], check=True)
    
    if models_dir is None:
        subprocess.run(['python', r'pipeline\_6_latent_nn.py'], check=True)
    
    subprocess.run(['python', r'pipeline\_7_latent_nn_pred.py'], check=True)

