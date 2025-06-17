import time
import asyncio
from typing import Any, Final
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig

from ai_common import (
    WebSearch,
    format_sources,
    get_config_from_runnable,
    NodeBase,
)


SUMMARIZER_INSTRUCTIONS = """
You are a world class researcher who is working on a report about a specific topic.

<goal>
Generate a very high quality informative summary of the given context in accordance with the topic.
</goal>

The topic you are working on:
<topic>
{topic}
</topic>

The context to use in generating the informative summary:
<context>
{context}
</context>

Prepare your summary according to the topic. 
Include all necessary information related with the topic in your summary.
"""


class WebSearchNode:
    def __init__(self,
                 web_search_api_key: str,
                 model_params: dict[str, Any],
                 configuration_module_prefix: str):
        self.web_search = WebSearch(api_key=web_search_api_key)
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.model_name = model_params['model']
        self.base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
           **model_params['model_args']
        )

    async def summarize_source(self, topic: str, source_dict: dict[str, Any]) -> (str, str, dict[str, Any]):
        max_length = 102400  # 100K
        raw_content = source_dict['raw_content'][:max_length] if source_dict['raw_content'] is not None else source_dict['content']
        instructions = SUMMARIZER_INSTRUCTIONS.format(topic=topic, context=raw_content)

        with get_usage_metadata_callback() as cb:
            summary = await self.base_llm.ainvoke(instructions)
            token_usage = {
                'input_tokens': cb.usage_metadata[self.model_name]['input_tokens'],
                'output_tokens': cb.usage_metadata[self.model_name]['output_tokens'],
            }

        return {
            'content': summary.content,
            'token_usage': token_usage
        }

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        event_loop = asyncio.get_event_loop()
        state = event_loop.run_until_complete(self.run_async(state=state, config=config))
        return state


    async def run_async(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        """
        Execute web searches, summarize content using LLM, and compile formatted results.
        
        This method performs web searches using the provided search queries from the state,
        retrieves unique sources, summarizes each source's content using the configured LLM,
        formats the summarized results, and updates the state with both formatted source 
        strings and processed source data for further processing.
        
        Args:
            state (BaseModel): The current flow state containing search queries and topic.
                             Must have 'search_queries' attribute with query objects
                             that have 'search_query' attributes, and 'topic' attribute
                             for summarization context.
            config (RunnableConfig): The configuration object
        
        Returns:
            BaseModel: The updated state with the following new attributes:
                      - source_str: Formatted string of summarized search results
                      - unique_sources: Source data with LLM-generated summaries
                      - steps: Updated with WEB_SEARCH node tracking
                      - token_usage: Updated with LLM token consumption
        
        Note:
            The method uses the configured search parameters (category, days back,
            max tokens per source, max results per query) to control search behavior.
            Each source is summarized using the base LLM with truncated content (max 100K chars).
        """
        if not hasattr(state, 'search_queries'):
            raise AttributeError("State must have a 'search_queries' attribute")
        
        if not state.search_queries:
            raise ValueError("State must contain at least one search query")
        
        for i, query in enumerate(state.search_queries):
            if not hasattr(query, 'search_query'):
                raise AttributeError(f"Query at index {i} must have a 'search_query' attribute")

        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        unique_sources = await self.web_search.search(
            search_queries = [query.search_query for query in state.search_queries],
            search_category = configurable.search_category,
            number_of_days_back = configurable.number_of_days_back,
            max_results_per_query = configurable.max_results_per_query,
        )

        tasks = [self.summarize_source(topic=state.topic, source_dict=v) for v in unique_sources.values()]
        out = await asyncio.gather(*tasks)

        state.token_usage[self.model_name]['input_tokens'] += sum([t['token_usage']['input_tokens'] for t in out])
        state.token_usage[self.model_name]['output_tokens'] += sum([t['token_usage']['output_tokens'] for t in out])

        unique_sources = {
            url: {
                'title': value['title'],
                'content': summary['content']
            }
            for url, value, summary in zip(unique_sources.keys(), unique_sources.values(), out)
        }

        source_str = format_sources(unique_sources=unique_sources,
                                    max_tokens_per_source=configurable.max_tokens_per_source,
                                    include_raw_content=False)
        state.steps.append(NodeBase.WEB_SEARCH.value)
        state.source_str = source_str
        state.unique_sources = unique_sources
        return state
