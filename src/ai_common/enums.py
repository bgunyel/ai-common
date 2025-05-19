from enum import Enum

class LlmServers(Enum): # Alphabetical Order
    GROQ = 'groq'
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    VLLM = 'vllm'
