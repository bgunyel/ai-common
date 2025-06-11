from enum import Enum

class LlmServers(Enum): # Alphabetical Order
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    VLLM = 'vllm'

class NodeBase(Enum):
    # In alphabetical order
    QUERY_WRITER = 'query_writer'
    WEB_SEARCH = 'web_search'
