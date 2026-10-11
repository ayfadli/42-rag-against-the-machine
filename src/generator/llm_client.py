import requests
import json
import os

def ask_llm(prompt: str) -> str:
    
    url = "https://localhost:11434/v1/chat/completions"

    headers = {
        'Content-type': 'application/json'
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
    

