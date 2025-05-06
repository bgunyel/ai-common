import asyncio
from tavily import AsyncTavilyClient

from .base import TavilySearchCategory
from .utils import tavily_search_async, deduplicate_and_format_sources, format_sources, deduplicate_sources


class WebSearch:
    def __init__(self,
                 api_key: str,
                 search_category: TavilySearchCategory,
                 number_of_days_back: int,
                 max_tokens_per_source: int = 5000,
                 include_raw_content: bool = False) -> None:
        self.event_loop = asyncio.get_event_loop()
        self.api_key = api_key
        self.search_category = search_category
        self.number_of_days_back = number_of_days_back
        self.max_tokens_per_source = max_tokens_per_source
        self.include_raw_content = include_raw_content

    def search(self, search_queries: list[str]) -> dict:

        search_docs = self.event_loop.run_until_complete(
            tavily_search_async(
                client=AsyncTavilyClient(api_key=self.api_key),
                search_queries=search_queries,
                search_category=self.search_category,
                number_of_days_back=self.number_of_days_back
            )
        )

        unique_sources = deduplicate_sources(search_response=search_docs)
        """
        source_str = deduplicate_and_format_sources(search_response=search_docs,
                                                    max_tokens_per_source=self.max_tokens_per_source,
                                                    include_raw_content=self.include_raw_content)
        """

        return unique_sources