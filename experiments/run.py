import glb
import os
from run_pipeline import run_pipeline
from experiments.retrieval_quality._1_get_data import save_run as retreival_quality_save_run
from experiments.generation_quality._1_get_data import save_run as generation_quality_save_run

start = 3
end = 4

for i in range(start-1, end):
    models_backup_dir = os.path.join(glb.cloud_dir, 'run{}'.format(i))
    print('run {} started'.format(i+1))
    test_case_seed = glb.test_case_seeds[i]
    run_pipeline(test_case_seed)
    glb.copy_files_from_dir(glb.models_dir, models_backup_dir)
    print('copy complete')
    print('run {} ended'.format(i+1))

start = 1
end = 10
for i in range(start-1, end):
    models_backup_dir = os.path.join(glb.cloud_dir, 'run{}'.format(i))
    print('run {} started'.format(i+1))
    test_case_seed = glb.test_case_seeds[i]
    run_pipeline(test_case_seed, models_dir=models_backup_dir)
    retreival_quality_save_run(i)
    print('retrieval complete')
    #generation_quality_save_run(i)
    #print('generation complete')
    print('run {} ended'.format(i+1))
       