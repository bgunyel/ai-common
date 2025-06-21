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


class NodeBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Class attributes
    QUERY_WRITER: ClassVar[str] = 'query_writer'
    WEB_SEARCH: ClassVar[str] = 'web_search'
