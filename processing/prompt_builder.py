import glb

class PromptBuilder():
    
    def __init__(self, query_builder):
        self.new_case = None
        self.query_builder = query_builder
        
    def set_new_case(self, new_case):
        self.new_case = new_case
        self.query_builder.set_new_case(new_case)
        
     
    def _get_context(self, classes):
        
        if classes is None:
            return (
                    'START CONTEXT'
                    '\n'
                    'To help you answer the question, below is a tip that may help:'
                    '\n'
                    'a) use a pair of numbers to make 24, 12, 8 or 6'
                    '\n'
                    'b) use the remaining pair of numbers to make 1, 2, 3 or 4 respectively'
                    '\n'
                    'c) then the product of step a) and step b) equals 24'
                    '\n'
                    'END CONTEXT'
                    )
        else:
            return (
                    'START CONTEXT'
                    '\n'
                    'To help you answer the question, below is a tip that may help:'
                    '\n'
                    '{}'
                    '\n'
                    'END CONTEXT'
                    ).format(
                        self._get_context_classes(classes)
                )
              

    def _get_context_class(self, _class):
         problem = self.new_case['problem_from_img']
         class_data = glb.classes[_class]
         num = class_data[0]
         other_num = int(24/class_data[0])
         pair_indexes = (class_data[1], class_data[2])
         pair = (problem[pair_indexes[0]], problem[pair_indexes[1]])
         return(
              'a) use the pair {} to make {}. If this is impossible, try make {} using some other pair'
              '\n'
              'b) then use the remaining pair to make {}'
              '\n'
              'c) then {} * {} = 24'
              ).format(
                  pair,
                  num,
                  num,
                  other_num,
                  num,
                  other_num
                  )      
                
            
    def _get_context_classes(self, classes):
        _str = ''
        for i, _class in enumerate(classes):
                _str += self._get_context_class(_class)
                if i < len(classes)-1:
                    _str += '\nOR\n'
                
        return _str
      
    
    def _get_query(self):
        query = self.query_builder.get_query()
        return (
            'START QUESTION'
            '\n'
            '{}'
            '\n'
            'END QUESTION'
            ).format(query)
        
    
    def get_role(self):
        return (
            'You are a student taking a test to solve a Math-24 puzzle.'
            '\n'
            'A Math-24 puzzle requires you to use 4 numbers to make 24. Each number must be used exactly once, and may use the + - * / operators.'
            '\n'
            'Once you have solved the puzzle, you must end your answer with "Final Answer: [LHS] = 24" where [LHS] is uses ALL 4 numbers EXACTLY once to get to 24.'
            '\n'
            'For example, if the puzzle is 1 2 9 13 then ending with "Final Answer: (13 + 9 + 2) * 1 = 24" will get you full marks.' 
            '\n'
            'If you end your answer using any other convention you will get no marks, even if your final answer is correct.'
            '\n'
            'For example, if you end with "Final Answer: (13+9+2)=24; 24*1=24", this will get you no marks because your format is wrong, even though your answer is correct.'
            '\n'
            'If you end with "Final Answer: (13 + 9 + 2) = 24", this will also get you no marks because you have omitted the 1, even though your answer is correct.'
            '\n'
            'When giving your final answer, do not use any special formatting such as bold or italics, latex, etc. You must use only plain text.'
            )

    
    def get_prompt(self, context_classes=None, get_context_if_no_class=True):
        if context_classes is None and get_context_if_no_class == False:
            return (
                '{}'
                ).format(
                    self._get_query(),
                    )
        else:    
            return (
                '{}'
                '\n\n'
                '{}'
                ).format(
                    self._get_query(),
                    self._get_context(classes=context_classes)
                    )