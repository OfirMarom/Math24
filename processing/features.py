import glb
from itertools import combinations, product, permutations
from sympy import sympify

class Features:
    
    all_numbers = {1, 2, 3, 4, 6, 8, 12, 24}
    
    def __init__(self, _case):
        self._case = _case
        
    def set_features(self):
        self._case['features'] = []
        self._case['features_scaled'] = []
        self._agg_by_group(n=2)
        self._agg_by_number(0)
        self._agg_by_number(1)
        self._agg_by_number(2)
        self._agg_by_number(3)
         
    def _get_expression(self, numbers, operators):
        expression = str(numbers[0])
        for i in range(len(operators)):
            expression += operators[i] + str(numbers[i+1]) 
        return expression   
    
    def _agg_by_number(self, index):
       counts = {num: 0 for num in Features.all_numbers}
       problem = self._case['problem']
       num1 = problem[index]
       op_combins = list(product(glb.operators, repeat=1))
       for i in range(len(problem)):
           if i == index:
               continue
           results = set()
           num2 = problem[i]
           for op_combin in op_combins:
               for perm in [[num1, num2], [num2, num1]]:
                   expression = self._get_expression(perm, op_combin)
                   result=float(sympify(expression))
                   results.add(result)
           for result in results:
               if result in counts:
                    counts[result] += 1   
       self._case['features'] += [count for num, count in counts.items()]
       scale = len(problem)-1
       self._case['features_scaled'] += [count /scale for num, count in counts.items()]
                
    def _agg_by_group(self, n, agg_by_number=True, agg_by_number_operator=True):
        counts = {num: 0 for num in Features.all_numbers}
        problem = self._case['problem']
        num_combins = list(combinations(problem, n))
        op_combins = list(product(glb.operators, repeat=n-1))
        for num_combin in num_combins:
            results = set()
            combin_perms = list(permutations(num_combin))
            for op_combin in op_combins:
                for combin_perm in combin_perms:
                    expression = self._get_expression(combin_perm, op_combin)
                    result=float(sympify(expression))
                    results.add(result)     
            for result in results:
                if result in counts:
                    counts[result] += 1
        self._case['features'] += [count for num, count in counts.items()]
        scale = len(num_combins)   
        self._case['features_scaled'] +=  [count /scale for num, count in counts.items()]
        
