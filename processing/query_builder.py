

class QueryBuilder():
    
    def __init__(self):
       self.new_case = None
    
    def set_new_case(self, new_case):
        self.new_case = new_case
          
    def get_query(self):
        problem = self.new_case['problem_from_img']
        query = (
            'Solve the following Math-24 puzzle:'
            '\n'
            '{} {} {} {}'
            ).format(
                problem[0],
                problem[1],
                problem[2],
                problem[3],
                ) 
        return query        
        