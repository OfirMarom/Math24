import glb
import os
import re
from sympy import sympify, SympifyError
from IPython import get_ipython

def find_final_answer(text):
    matches = list(re.finditer('final answer', text, re.IGNORECASE))
    if not matches:
        return None
    last_index = matches[-1].start()
    final_answer = text[last_index:]
    _match = re.search(r'^final answer\s*:\s*([\s\S]*?)=\s*24', final_answer, re.IGNORECASE)
    if not _match:
        return None
    return _match.group(1)
    
def get_n_equal_signs(text):
    return len(text.split('='))-1
    
def check_numbers(text, problem):
    numbers = re.findall(r'\d+', text)
    numbers = [int(val) for val in numbers]
    return sorted(numbers) == problem

def clean_text(text):
    text = text.replace('\*','*')
    text = text.replace('×','*')
    text = text.replace('÷','/')
    text = text.replace('[','(')
    text = text.replace(']',')')
    return text
    
def val_response(response, problem):
    success = None
    val = None
    final_answer = find_final_answer(response)
    if final_answer is None:
        success = 0
    else:
        final_answer = clean_text(final_answer)
        n_equal_signs = get_n_equal_signs(final_answer)
        if n_equal_signs > 0:
            success = None
        else:    
            numbers_valid = check_numbers(final_answer, problem)
            if not numbers_valid:
                success = False    
            else:    
                try:
                    val = sympify(final_answer)
                except SympifyError:
                    success=None
                if val is not None:    
                    if val == 24:
                        success=True
                    else: 
                        success=False
    return success

filenames = os.listdir(glb.results_dir)
filenames = [filename for filename in filenames if filename.startswith('llm-')]

for filename in filenames:

    path = os.path.join(glb.results_dir, filename)
    results = glb.open_json(path)
    
    for result in results:
        if result is None:
            continue
        for test_case_result in result:
            for prompt_type in test_case_result['responses']:
                d = test_case_result['responses'][prompt_type]
                if d['success'] is not None:
                    continue
                success = val_response(d['response'], test_case_result['test_problem'])   
                if success == True:
                    d['success']=1
                elif success == False:
                    d['success']=0
                else:
                    get_ipython().magic('clear') 
                    print('could not automatically classify')
                    print('response: {}'.format(d['response']))
                    print('problem: {}'.format(test_case_result['test_problem']))
                    print('solutions: {}'.format(test_case_result['test_solutions']))
                    correct = input('Enter 1 for correct, 0 for fail: ')
                    d['success']= int(correct)

    glb.save_json(results, path)
    
            
  