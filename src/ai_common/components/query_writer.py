import asyncio
import datetime
import json
from typing import Any, Final

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel

from ai_common import get_config_from_runnable, NodeBase, SearchQuery

QUERY_WRITER_INSTRUCTIONS = """
<Goal>
Your goal is to generate targeted web search queries that will gather comprehensive information for writing a summary about a topic.
You will generate exactly {number_of_queries} queries.
</Goal>

<topic>
{topic}
</topic>

Today's date is:
<today>
{today}
</today>

<Requirements>
When generating the search queries:
1. Make sure to cover different aspects of the topic.
2. Make sure that your queries account for the most current information available as of today.

Your queries should be:
- Specific enough to avoid generic or irrelevant results.
- Targeted to gather specific information about the topic.
- Diverse enough to cover all aspects of the summary plan.
</Requirements>

<Format>
* Format your response as a JSON object with one field:
    - queries: Queries you generate according to the given topic.
* Each query should have the following three fields:
    - search_query: Text of the query.
    - aspect: Which aspect of the topic the query aims to cover.
    - rationale: Your reasoning.    

Return the queries in JSON format:
{{
    queries: [
            {{
                "search_query": "string",
                "aspect": "string",
                "rationale": "string"
            }}
    ]
}}
</Format>

<Task>
It is very important that you generate exactly {number_of_queries} queries.
Generate targeted web search queries that will gather specific information about the given topic.
</Task>
"""


class QueryWriter:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:

        if not hasattr(state, 'topic'):
            raise AttributeError("State must have a 'topic' attribute")
        
        configurable = get_config_from_runnable(
            configuration_module_prefix = self.configuration_module_prefix,
            config = config
        )
        state.steps.append(NodeBase.QUERY_WRITER)
        """
        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format = {"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            json_dict = json.loads(results.content)
        """

        ## TODO
        raise NotImplementedError('This shall be parallelized (in accordance with question-tracing in SummaryWriter)')
        state.search_queries = [SearchQuery(**q) for q in json_dict['queries']]
        return state


    async def generate_queries(self, topic: str, number_of_queries: int):
        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = await self.base_llm.ainvoke(instructions, response_format={"type": "json_object"})
            token_usage = {
                'input_tokens': cb.usage_metadata[self.model_name]['input_tokens'],
                'output_tokens': cb.usage_metadata[self.model_name]['output_tokens'],
            }

            json_dict = json.loads(results.content)
            search_queries = [SearchQuery(**q) for q in json_dict['queries']]

        return {
            'search_queries': search_queries,
            'token_usage': token_usage,
        }
