from enum import Enum
from typing import ClassVar
from pydantic import BaseModel, ConfigDict

class LlmServers(Enum): # Alphabetical Order
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OPENAI = 'openai'
    OLLAMA = 'ollama'
    VLLM = 'vllm'

class ModelNames(Enum): # Most used model names in alphabetical order
    """
    * Same model can be served under different names on different platforms
    *
    * e.g. gpt-oss-120b is named as:
        * openai/gpt-oss-120b on Groq
        * gpt-oss:120b-cloud on Ollama Cloud
    """
    GPT_OSS_120B = 'gpt-oss-120b'
    GPT_OSS_20B = 'gpt-oss-20b'
    LLAMA_3_3_70B_VERSATILE = 'llama-3.3-70b-versatile'


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
