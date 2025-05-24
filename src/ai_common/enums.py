from enum import Enum

class LlmServers(Enum): # Alphabetical Order
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    VLLM = 'vllm'
