import requests
import json
import os
from dotenv import load_dotenv

def ask_llm(prompt: str, model: str = "Qwen/Qwen3-0.6B") -> str:
    
    load_dotenv()
 
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environement variable is missing."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        'Content-type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        'model': 'qwen2.5',
        'messages': [
        {'role': "user"}, {"content": "prompt"}
        ],
        'temperature': 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content']
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is the server running? (Run 'ollama serve')"
    except Exception as e:
        return f"API Call Failed: {e}"
    

