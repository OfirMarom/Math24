import glb
import os
from processing.query_builder import QueryBuilder
from processing.prompt_builder import PromptBuilder
from processing.similar_cases import SimilarCases
from llm_api import LLMAPI
import time

llm_api = LLMAPI()

llm_provider = glb.LLMProviders.together_ai
models = [
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'meta-llama/Llama-3-70b-chat-hf',
    'Qwen/Qwen2-72B-Instruct',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'databricks/dbrx-instruct',  
    ]   
  
def save_run(run_index):

    query_builder = QueryBuilder()
    prompt_builder = PromptBuilder(query_builder)
    role = prompt_builder.get_role()
    cases = glb.open_json(os.path.join(glb.data_dir, 'cases.json'))    
    test_cases = [_case for _case in cases if _case['is_test']==True]       

    for model in models:
        model_name = model.split('/')[-1]
        print('model {}'.format(model))
        
        path = os.path.join(glb.results_dir, 'llm-{}.json'.format(model_name))
        if os.path.exists(path):
            results = glb.open_json(path)
        else:        
            results = [None] * len(glb.test_case_seeds)
        result = []    
        for j, test_case in enumerate(test_cases):
            print('test case {}'.format(j))
            similar_cases = SimilarCases(test_case)
            sim_case = similar_cases.get(1, 'latent')[0]
            sim_case_classes = sim_case['classes']
            prompt_builder.set_new_case(test_case)
            prompt_no_context = prompt_builder.get_prompt(get_context_if_no_class=False)
            prompt_context = prompt_builder.get_prompt(get_context_if_no_class=True)
            prompt_rag = prompt_builder.get_prompt(context_classes=sim_case_classes)  
            test_case_result = {
                'test_problem': test_case['problem_from_img'],
                'sim_problem': sim_case['problem_from_img'],
                'test_case_classes': test_case['classes'],
                'sim_case_classes': sim_case_classes,
                'test_solutions': test_case['solutions'],
                'prompts': { 'no_context': prompt_no_context, 'context': prompt_context, 'rag': prompt_rag},
                'responses': {}
                }
            for prompt_type, prompt in test_case_result['prompts'].items():
                if prompt_type not in test_case_result['responses']:
                    test_case_result['responses'][prompt_type] = {}
                if llm_provider == glb.LLMProviders.groq and glb.count_llm_queries == 30:
                    time.sleep(120)
                    glb.count_llm_queries = 0
                if llm_provider == glb.LLMProviders.together_ai and glb.count_llm_queries == 600:
                    time.sleep(120)
                    glb.count_llm_queries = 0        
                time.sleep(0.1)    
                response = llm_api.get_response(llm_provider, model, role, prompt)
                test_case_result['responses'][prompt_type] = { 'response': response, 'success' : None }
                glb.count_llm_queries+=1
            result.append(test_case_result) 
            
        results[run_index] = result
        glb.save_json(results, path)
      