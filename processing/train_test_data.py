import glb
import numpy as np

class TrainTestData:
    
    def __init__(self, cases):
        self.cases = cases
        self.X = None
        self.Y = None
    
                    
    def set_from_cases(self):
        
        n_problem = len(self.cases[0]['problem'])
        n_feature = len(self.cases[0]['features_scaled'])
        
        
        self.X = np.zeros((len(self.cases), n_problem + n_feature), dtype=float) 
        for i, _case in enumerate(self.cases):
           self.X[i,0: n_problem] = [(v-1)/ (glb.max_problem_num-1) for v in _case['problem_from_img']] 
           self.X[i,n_problem:] = _case['features_scaled']
       
        self.Y = np.zeros((len(self.cases), 20), dtype=int)
        
        for i, _case in enumerate(self.cases):
            
            for _class in _case['classes']:
                
                class_data = glb.classes[_class]
                
                if class_data[0]==24:
                    start=0
                elif class_data[0]==12:
                    start=5
                elif class_data[0]==8:
                    start=10
                elif class_data[0]==6:
                    start=15
                
                if class_data[1]==0:
                    mid = start+1
                elif class_data[1]==1:
                    mid = start+2
                elif class_data[1]==2:
                    mid = start+3
        
                if class_data[2]==1:
                    end = start+2
                elif class_data[2]==2:
                    end = start+3
                elif class_data[2]==3:
                    end = start+4  
                    
                self.Y[i, [start, mid, end]]=1
    

                 
    def get(self):
        return self.X, self.Y      
    
   