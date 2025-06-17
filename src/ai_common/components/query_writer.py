import datetime
from typing import Any, Final
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel

from ai_common import Queries, get_config_from_runnable, NodeBase

QUERY_WRITER_INSTRUCTIONS = """
Your goal is to generate targeted web search queries that will gather comprehensive information for writing a summary about a topic.
You will generate exactly {number_of_queries} queries.

<topic>
{topic}
</topic>

Today's date is:
<today>
{today}
</today>

When generating the search queries:
1. Make sure to cover different aspects of the topic.
2. Make sure that your queries account for the most current information available as of today.

Your queries should be:
- Specific enough to avoid generic or irrelevant results.
- Targeted to gather specific information about the topic.
- Diverse enough to cover all aspects of the summary plan.

It is very important that you generate exactly {number_of_queries} queries.
Generate targeted web search queries that will gather specific information about the given topic.
"""


class QueryWriter:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        base_llm = init_chat_model(kwargs=model_params)
        self.structured_llm = base_llm.with_structured_output(Queries)

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        """
        Generate targeted web search queries using LLM with structured output.
        
        This method creates a specified number of diverse, specific search queries
        designed to gather comprehensive information about a given topic. The queries
        are generated using an LLM with structured output (Queries schema) to ensure 
        they cover different aspects of the topic and account for current information.
        
        Args:
            state (BaseModel): The current flow state containing the research topic.
                             Must have a 'topic' attribute.
            config (RunnableConfig): The runnable configuration containing parameters
                                   like the number of queries to generate.
        
        Returns:
            BaseModel: The updated state with the following updates:
                      - search_queries: List of generated Query objects with search_query attributes
                      - steps: Updated with QUERY_WRITER node tracking
                      - token_usage: Updated with LLM token consumption
        
        Note:
            Uses structured LLM output to ensure consistent query format and tracks
            token usage for cost monitoring. Includes today's date in instructions
            to generate queries that capture current information.
        """
        if not hasattr(state, 'topic'):
            raise AttributeError("State must have a 'topic' attribute")
        
        configurable = get_config_from_runnable(
            configuration_module_prefix = self.configuration_module_prefix,
            config = config
        )
        state.steps.append(NodeBase.QUERY_WRITER.value)
        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = self.structured_llm.invoke(instructions)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
        state.search_queries = results.queries
        return state
