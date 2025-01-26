from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def list_vision_models():
    client = OpenAI()
    models = client.models.list()
    
    print('Available Vision Models:')
    vision_models = [model.id for model in models.data if 'vision' in model.id.lower()]
    
    if vision_models:
        for model in vision_models:
            print(f'- {model}')
    else:
        print('No vision models found')

if __name__ == "__main__":
    list_vision_models()
