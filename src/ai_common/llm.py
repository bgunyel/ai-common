from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from ollama import Client

from .enums import LlmServers
from .tools import _check_and_pull_ollama_model


def load_ollama_model(model_name: str, ollama_url: str) -> None:
    _check_and_pull_ollama_model(model_name=model_name, ollama_url=ollama_url)
    ollama_client = Client(host=ollama_url)
    ollama_client.generate(model=model_name)  # Generate w/ prompt loads the model to memory


def get_llm(llm_server: LlmServers, model_params: dict[str, Any]) -> BaseChatModel:
    llm_base_url = model_params.get('llm_base_url', '')

    match llm_server:
        case LlmServers.ANTHROPIC:
            llm = ChatAnthropic(
                model_name=model_params['model_name'],
                temperature=0,
                api_key=model_params['anthropic_api_key'],
                timeout=model_params['default_request_timeout'],
                stop=model_params['stop_sequences']
            )
        case LlmServers.GROQ:
            llm = ChatGroq(
                model=model_params['model_name'],
                temperature=0,
                api_key=model_params['groq_api_key'],
            )
        case LlmServers.OPENAI:
            llm = ChatOpenAI(
                model=model_params['model_name'],
                temperature=0,
                api_key=model_params['openai_api_key'],
            )
        case LlmServers.OLLAMA:
            llm = ChatOllama(
                model=model_params['model_name'],
                temperature=0,
                base_url=llm_base_url,
                format=model_params['format'],
                num_ctx=model_params['context_window_length'],
            )
        case LlmServers.VLLM:
            client = OpenAI(base_url=f'{llm_base_url}/v1', api_key=model_params['vllm_api_key'])
            # max_model_len = client.models.list().data[0].model_extra['max_model_len']  # Keep this
            llm = ChatOpenAI(api_key=model_params['vllm_api_key'],
                             model=client.models.list().data[0].id,
                             temperature=0,
                             base_url=f'{llm_base_url}/v1')
        case _:
            raise ValueError(f'LLM Server {llm_server.value} is currently not supported!')
    return llm
