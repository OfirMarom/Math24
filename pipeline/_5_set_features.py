import glb
import os
from processing.features import Features


cases = glb.open_json(os.path.join(glb.data_dir, 'cases.json'))
     
for _case in cases:
    problem = _case['problem']
    features = Features(_case)
    features.set_features()

glb.save_json(cases, os.path.join(glb.data_dir, 'cases.json'))
