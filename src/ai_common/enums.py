from enum import Enum
from typing import ClassVar
from pydantic import BaseModel, ConfigDict

class LlmServers(Enum): # Alphabetical Order
    ANTHROPIC = 'anthropic'
    GOOGLE = 'google'
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
    DEEPSEEK_V_3_2 = 'deepseek-v3.2'
    GLM_5 = 'glm-5'
    GPT_5 = 'gpt-5'
    GPT_5_MINI = 'gpt-5-mini'
    GPT_5_1 = 'gpt-5.1'
    GPT_5_2 = 'gpt-5.2'
    GPT_OSS_120B = 'gpt-oss-120b'
    GPT_OSS_20B = 'gpt-oss-20b'
    KIMI_K2_0905 = 'kimi-k2-instruct-0905'
    KIMI_K_2_5 = 'kimi-k2.5'
    LLAMA_3_3_70B_VERSATILE = 'llama-3.3-70b-versatile'
    MINIMAX_M_2_5 = 'minimax-m2.5'
    MINIMAX_M_2_7 = 'minimax-m2.7'
    NEMOTRON_3_SUPER = 'nemotron-3-super'
    QWEN_3_5 = 'qwen-3.5'

    # Update Gemini-3 Family,
    GEMINI_3_FLASH_PREVIEW = 'gemini-3-flash-preview'
    GEMINI_3_1_PRO_PREVIEW = 'gemini-3.1-pro-preview'
    GEMINI_3_1_FLASH_LITE_PREVIEW = 'gemini-3.1-flash-lite-preview'


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
