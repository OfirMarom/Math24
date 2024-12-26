from PIL import Image

class ProblemToImage:
    
    def __init__(self, problem):
        self.problem = problem
        self.img = None
        
    def set_img(self):
        num_img1 = Image.open('assets/{}.png'.format(self.problem[0])).convert('L') 
        num_img2 = Image.open('assets/{}.png'.format(self.problem[1])).convert('L') 
        num_img3 = Image.open('assets/{}.png'.format(self.problem[2])).convert('L') 
        num_img4 = Image.open('assets/{}.png'.format(self.problem[3])).convert('L') 
        num_width, num_height = num_img1.size
        canvas = Image.new('L', (num_width * 2, num_height * 2))     
        canvas.paste(num_img1, (0, 0)) 
        canvas.paste(num_img2, (num_width, 0))  
        canvas.paste(num_img3, (0, num_height))  
        canvas.paste(num_img4, (num_width, num_height))  
        self.img = canvas
        
    
    def save(self, path):
        self.img.save(path)
        
        