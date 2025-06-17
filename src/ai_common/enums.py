from enum import Enum

class LlmServers(Enum): # Alphabetical Order
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OPENAI = 'openai'
    # Self-serve
    OLLAMA = 'ollama'
    VLLM = 'vllm'

class NodeBase(Enum):
    # In alphabetical order
    QUERY_WRITER = 'query_writer'
    WEB_SEARCH = 'web_search'
