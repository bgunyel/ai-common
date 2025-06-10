import asyncio

from tavily import AsyncTavilyClient

from .base import TavilySearchCategory
from .utils import tavily_search_async, deduplicate_sources


class WebSearch:
    def __init__(self, api_key: str) -> None:
        self.event_loop = asyncio.get_event_loop()
        self.api_key = api_key

    def search(self,
               search_queries: list[str],
               search_category: TavilySearchCategory,
               number_of_days_back: int,
               max_results_per_query: int) -> dict:

        search_docs = self.event_loop.run_until_complete(
            tavily_search_async(
                client=AsyncTavilyClient(api_key=self.api_key),
                search_queries=search_queries,
                search_category=search_category,
                number_of_days_back=number_of_days_back,
                max_results=max_results_per_query,
            )
        )
        unique_sources = deduplicate_sources(search_response=search_docs)
        return unique_sources
