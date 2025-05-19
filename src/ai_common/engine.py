import os
import datetime
from typing import Any

from .base import GraphBase
from .enums import LlmServers
from .llm import load_ollama_model
from .utils import get_flow_chart


def save_response(response: str, save_to_folder: str):
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    file_name = os.path.join(save_to_folder, f'response-{time_now.isoformat()}.md')
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f'{response}')


class Engine:
    def __init__(self, responder: GraphBase, llm_server: LlmServers, models: list[str], llm_base_url: str, save_to_folder: str):

        if llm_server == LlmServers.OLLAMA:
            for model in models:
                load_ollama_model(model_name=model, ollama_url=f'{llm_base_url}')

        self.history = []
        self.responder = responder
        self.save_to_folder = save_to_folder

    def save_flow_chart(self, save_to_folder: str):
        flow_chart = get_flow_chart(rag_model=self.responder)
        flow_chart.save(os.path.join(save_to_folder, 'flow_chart.png'))

    def get_response(self, input_dict: dict[str, Any]):
        self.history.append({"role": "user", "content": input_dict})
        response = self.responder.get_response(input_dict=input_dict, verbose=False)
        self.history.append({"role": "assistant", "content": response})
        save_response(response=response, save_to_folder=self.save_to_folder)
        return response
