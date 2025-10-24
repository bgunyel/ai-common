from enum import Enum
from typing import ClassVar
from pydantic import BaseModel, ConfigDict

class LlmServers(Enum): # Alphabetical Order
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OPENAI = 'openai'
    # Self-serve
    OLLAMA = 'ollama'
    VLLM = 'vllm'

class TavilySearchCategory(Enum):
    # https://docs.tavily.com/documentation/api-reference/endpoint/search
    GENERAL = 'general'
    FINANCE = 'finance'
    NEWS = 'news'

class TavilySearchDepth(Enum):
    # https://docs.tavily.com/documentation/api-reference/endpoint/search
    ADVANCED = 'advanced'
    BASIC = 'basic'

class NodeBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Class attributes
    QUERY_WRITER: ClassVar[str] = 'query_writer'
    WEB_SEARCH: ClassVar[str] = 'web_search'
