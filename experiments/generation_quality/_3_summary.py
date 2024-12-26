import glb
import os
import numpy as np
import copy

filenames = os.listdir(glb.results_dir)
filenames = [filename for filename in filenames if filename.startswith('llm-')]

metrics = {
    'length': [],
    'accuracy': [],
    'faithfulness': [],
    'negative_rejection': []
    }

for filename in filenames:
    base_filename, _ = os.path.splitext(filename)
    model = base_filename[4:]
    scores = {}
    path = os.path.join(glb.results_dir, filename)
    results = glb.open_json(path)
    results = [result for result in results if result is not None]
    for result in results:
        run_scores = {}
        for test_case_result in result:
            is_context_correct = any(x in test_case_result['test_case_classes'] for x in test_case_result['sim_case_classes'])
            for prompt_type in test_case_result['responses']:
                if prompt_type not in scores:
                    scores[prompt_type] = copy.deepcopy(metrics)
                if prompt_type not in run_scores:
                    run_scores[prompt_type] = copy.deepcopy(metrics)
                d = test_case_result['responses'][prompt_type]
                length = len(d['response'])
                success = d['success']
                if is_context_correct:
                    run_scores[prompt_type]['faithfulness'].append(success) 
                else:
                    run_scores[prompt_type]['negative_rejection'].append(success) 
                run_scores[prompt_type]['length'].append(length)
                run_scores[prompt_type]['accuracy'].append(success)
        for prompt_type in run_scores:
             for metric in run_scores[prompt_type]:
                  vals = run_scores[prompt_type][metric]
                  m = np.mean(vals)
                  scores[prompt_type][metric].append(m)  
    for prompt_type in scores:
          for metric in scores[prompt_type]:              
             mean = np.mean(scores[prompt_type][metric])
             print('{} {} {} {}'.format(model, prompt_type, metric, mean))
    print()
       
            
            
            
            
            
            
