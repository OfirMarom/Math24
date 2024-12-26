import json
from sklearn.metrics.pairwise import cosine_similarity

class SimilarCases:
    
    def __init__(self, new_case):
        self.new_case = new_case
        self.solved_cases = None
        self._load_solved_cases()
        
    def _load_solved_cases(self):        
        with open('gitignore/cases.json', 'r') as file:
            cases = json.load(file)
        self.solved_cases = [_case for _case in cases if _case['is_test']==False]

    def _get_cosine_sims(self, method):
        latent = self.new_case[method]
        solved_latents = [_case[method] for _case in self.solved_cases]
        sims = cosine_similarity([latent], solved_latents)[0]
        d_cosine_sim = {}
        for i in range(len(solved_latents)):
            d_cosine_sim[i] = sims[i]
        d_cosine_sim = dict(sorted(d_cosine_sim.items(), key=lambda item: -item[1]))
        return d_cosine_sim

    def get(self, top_k, method):
        d_cosine_sim = self._get_cosine_sims(method)
        top_k_indexes = list(d_cosine_sim.keys())[0:top_k]
        sim_cases = [self.solved_cases[i] for i in top_k_indexes]
        return sim_cases
