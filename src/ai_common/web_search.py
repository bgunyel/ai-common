import asyncio

from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from .base import ConfigurationBase, StateBase, TavilySearchCategory
from .utils import tavily_search_async, deduplicate_and_format_sources


class WebSearch:
    def __init__(self, api_key: str, search_category: TavilySearchCategory, number_of_days_back: int) -> None:
        self.event_loop = asyncio.get_event_loop()
        self.api_key = api_key
        self.search_category = search_category
        self.number_of_days_back = number_of_days_back

    def run(self, state: StateBase, config: RunnableConfig) -> StateBase:

        search_docs = self.event_loop.run_until_complete(
            tavily_search_async(
                client=AsyncTavilyClient(api_key=self.api_key),
                search_queries=state.search_queries,
                search_category=self.search_category,
                number_of_days_back=self.number_of_days_back
            )
        )

        source_str = deduplicate_and_format_sources(search_response=search_docs,
                                                    max_tokens_per_source=5000,
                                                    include_raw_content=True)

        state.steps.append('web_search')
        state.source_str = source_str

        return state
