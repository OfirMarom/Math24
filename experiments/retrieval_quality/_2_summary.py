import glb
import os
import numpy as np
       
path = os.path.join(glb.results_dir, 'retrieval_data.json')

results = glb.open_json(path)    
results = [result for result in results if result is not None]    

summary = {}

for result in results:
    for label_method in result:
        if label_method not in summary:
            summary[label_method] = {}
        for k in result[label_method]:
            if k not in summary[label_method]:
                summary[label_method][k] = {}
            for sim_method in result[label_method][k]:
                if sim_method not in summary[label_method][k]:
                    summary[label_method][k][sim_method] = {key: [result[label_method][k][sim_method][key]] for key in result[label_method][k][sim_method]}
                else:
                    summary[label_method][k][sim_method] = {key: summary[label_method][k][sim_method][key] + [result[label_method][k][sim_method][key]] for key in result[label_method][k][sim_method]} 
              
summary2 = {}
            
print('Mean')        
for label_method in summary:
    if label_method not in summary2:
        summary2[label_method] = {}
    for k in summary[label_method]:
        for sim_method in summary[label_method][k]:
            if sim_method not in summary2[label_method]:
                summary2[label_method][sim_method] = {}
            for metric in summary[label_method][k][sim_method]:
                if metric not in summary2[label_method][sim_method]:
                    summary2[label_method][sim_method][metric] = []
                mean = np.mean(summary[label_method][k][sim_method][metric])
                summary2[label_method][sim_method][metric].append(mean)
                print('{} {} {} {} {}'.format(label_method, k, sim_method, metric, mean))
                
                
print('Mean2')                 
for label_method in summary2:
    for sim_method in summary2[label_method]:
        for metric in summary2[label_method][sim_method]:
            mean = np.mean(summary2[label_method][sim_method][metric])
            print('{} {} {} {}'.format(label_method, sim_method, metric, mean))
        
            
    

