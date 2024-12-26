import glb
import os
from groq import Groq
from together import Together


class LLMAPI:
     
    def __init__(self):
        self.temperature=0.7
        self.max_tokens=1024
        self.top_p=0.8
        self.top_k = 50
    
            
    def get_response(self, provider, model, system_content, user_content):
        if provider == glb.LLMProviders.groq:
            return self._get_groq_response(model, system_content, user_content)
        elif provider == glb.LLMProviders.together_ai:
            return self._get_together_ai_response(model, system_content, user_content)
                 
            
    def _get_groq_response(self, model, system_content, user_content):
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                { "role": "system", "content": system_content },   
                { "role": "user", "content": user_content },
            ],
            temperature= self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k
        )
        return response.choices[0].message.content
        
    
    def _get_together_ai_response(self, model, system_content, user_content):
        client = Together(
            api_key=os.getenv("TOGETHERAI_API_KEY"),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                { "role": "system", "content": system_content },   
                { "role": "user", "content": user_content },
            ],
            temperature= self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k
        )
        return response.choices[0].message.content    
