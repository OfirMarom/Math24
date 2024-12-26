import glb
import csv
import os
import re
from sympy import sympify
import random
from processing.problem_to_image import ProblemToImage
import uuid
import sys


def get_numbers(text):
    numbers = re.findall(r'\d+', text)
    numbers = [int(val) for val in numbers]
    return numbers

seed = int(sys.argv[1])  
random.seed(seed)

glb.delete_files_in_dir(glb.case_images_dir)
   
problem_pattern = r'(\(\d{1,2}[*+-/]\d{1,2}\)|\d{1,2}[*/]\d{1,2})\*(\(\d{1,2}[*+-/]\d{1,2}\)|\d{1,2}[*/]\d{1,2})'

cases = []
operators_pattern = '[' + ''.join(glb.operators) + ']'

with open('solutions.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    first = True
    for row in csv_reader:
        if first:
            first=False
            continue
        problem = [int(v) for v in row[1].split()]
        solutions = []
        for i in range(2,13):
            solution = row[i]
            if len(solution)==0:
                continue
            solution = solution.replace('×','*')
            solutions.append(solution)
            _case = {'problem': problem ,'solutions': solutions, 'is_test': False, 'problem_img_filename': None }
        cases.append(_case)
    
for i, _case in enumerate(cases):
    solutions = _case['solutions']
    classes = set() 
    for solution in solutions:
        problem = _case['problem'].copy()
        _match = re.search(problem_pattern, solution)
        if not _match:
            continue
        num_left = float(sympify(_match.group(1)))
        num_right = float(sympify(_match.group(2)))
        num_max = max(num_left, num_right)
        if num_left > num_right:
            pair = get_numbers(_match.group(1))
        else:
            pair = get_numbers(_match.group(2))
        pair_min = min(pair[0], pair[1])
        pair_max = max(pair[0], pair[1])
        pair_min_index = problem.index(pair_min)
        problem.pop(pair_min_index)
        pair_max_index = problem.index(pair_max) + 1
        class_data = (num_max, pair_min_index, pair_max_index)  
        if class_data not in glb.classes:
            continue
        _class = glb.classes.index(class_data)
        classes.add(_class)
            
    _case['classes'] = list(classes)    
            
success = 0   
for i, _case in enumerate(cases):
    if _case['classes']:
        success+=1
     
print('kept cases proportion: {}'.format(success/ len(cases)))
      
cases = [_case for _case in cases if _case['classes']]  

for _case in cases:
    problem = _case['problem'].copy()
    random.shuffle(problem)
    problem_to_image = ProblemToImage(_case['problem'])
    problem_to_image.set_img()
    filename = str(uuid.uuid4()) + '.png'
    img_save_path = os.path.join(glb.case_images_dir, filename)
    problem_to_image.save(img_save_path)
    _case['problem_img_filename'] = filename

test_cases = random.sample(cases, glb.n_test_cases)
for _case in test_cases:
    _case['is_test'] = True    
         
class_dist = {}
for _case in cases:
    if _case['is_test']:
        continue
    for _class in _case['classes']:
        if _class not in class_dist:
            class_dist[_class]=0
        class_dist[_class]+=1

class_dist = {k: class_dist[k] for k in sorted(class_dist)}

print('class distribution:') 
print(class_dist)

glb.save_json(cases, os.path.join(glb.data_dir, 'cases.json'))
