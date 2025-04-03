import requests
from langchain_ollama import OllamaLLM
import time
from requests.exceptions import RequestException

def get_available_models(max_retries=3, retry_delay=1):
    """
    Retrieve list of available models from Ollama service.
    
    Args:
        max_retries (int): Maximum connection attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        list: Available model names
        
    Raises:
        RequestException: If cannot connect to Ollama service
    """
    for attempt in range(max_retries):
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            response.raise_for_status()
            return [model['name'] for model in response.json()['models']]
        except RequestException as e:
            if attempt == max_retries - 1:
                print(f"Warning: Could not connect to Ollama service: {e}")
                return ['llama2', 'codellama']  # fallback defaults
            print(f"Retry {attempt + 1}/{max_retries} connecting to Ollama...")
            time.sleep(retry_delay)

def initialize_llm(model_name):
    """
    Initialize LLM interface with specified model.
    
    Args:
        model_name (str): Name of the model to initialize
        
    Returns:
        OllamaLLM: Initialized LLM interface
        
    Raises:
        Exception: If initialization fails
    """
    try:
        llm = OllamaLLM(
            model=model_name,
            temperature=0.7,
            request_timeout=30.0,  # Increase timeout
            base_url="http://localhost:11434",  # Explicitly set base URL
        )
        # Test the connection
        llm.invoke("test")
        return llm
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        print("Please ensure Ollama service is running: 'ollama serve'")
        raise
