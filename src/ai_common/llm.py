from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from ollama import Client
from pydantic import SecretStr

from .enums import LlmServers, ModelNames
from .tools import _check_and_pull_ollama_model


def load_ollama_model(model_name: str, ollama_url: str) -> None:
    _check_and_pull_ollama_model(model_name=model_name, ollama_url=ollama_url)
    ollama_client = Client(host=ollama_url)
    ollama_client.generate(model=model_name)  # Generate w/ prompt loads the model to memory


def get_llm(model_name: ModelNames,
            model_provider: LlmServers,
            api_key: SecretStr,
            model_args: dict[str, Any]) -> BaseChatModel:

    model_name_alias_dict = {
        ModelNames.GPT_OSS_120B.value: {
            LlmServers.GROQ.value: 'openai/gpt-oss-120b',
            LlmServers.OLLAMA.value: 'gpt-oss:120b-cloud'
        },
        ModelNames.GPT_OSS_20B.value: {
            LlmServers.GROQ.value: 'openai/gpt-oss-20b',
            LlmServers.OLLAMA.value: 'gpt-oss:20b-cloud'
        }
    }

    model_name_str = model_name_alias_dict[model_name.value][model_provider.value] if model_name.value in model_name_alias_dict.keys() else model_name.value

    llm = None
    match model_provider:
        case LlmServers.ANTHROPIC:
            llm = ChatAnthropic(
                model_name = model_name_str,
                api_key = api_key,
                stop = None,
                timeout = None,
                **model_args,
            )
        case LlmServers.GROQ:
            if 'top_p' in model_args.keys():
                model_args['model_kwargs'] = {
                    'top_p': model_args.pop('top_p')
                }
            if ('reasoning' in model_args.keys()) and ('reasoning_effort' not in model_args.keys()):
                model_args['reasoning_effort'] = model_args.pop('reasoning')

            llm = ChatGroq(
                model = model_name_str,
                api_key = api_key,
                service_tier = "auto",
                **model_args,
            )
        case LlmServers.OPENAI:
            llm = ChatOpenAI(
                model = model_name_str,
                api_key = api_key,
                **model_args,
            )
        case LlmServers.OLLAMA:

            if ('reasoning' not in model_args.keys()) and ('reasoning_effort' in model_args.keys()):
                model_args['reasoning'] = model_args.pop('reasoning_effort')

            llm = ChatOllama(
                model = model_name_str,
                client_kwargs={
                    'headers': {'Authorization': f'Bearer {api_key}'}
                },
                base_url = "https://ollama.com",
                **model_args,
            )
        case LlmServers.VLLM:
            # client = OpenAI(base_url=f'{llm_base_url}/v1', api_key=model_params['vllm_api_key'])
            # max_model_len = client.models.list().data[0].model_extra['max_model_len']  # Keep this
            # llm = ChatOpenAI(api_key=model_params['vllm_api_key'],
            #                 model=client.models.list().data[0].id,
            #                 temperature=0,
            #                 base_url=f'{llm_base_url}/v1')
            raise NotImplementedError(f'LLM Server {model_provider.value} is not yet implemented!')
        case _:
            raise ValueError(f'LLM Server {model_provider.value} is currently not supported!')
    return llm
