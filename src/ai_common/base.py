import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Optional, TypeAlias, Literal, List

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

TavilySearchCategory: TypeAlias = Literal['news', 'general']


class CfgBase(BaseModel):
    thread_id: str

    @classmethod
    def from_runnable(cls, runnable: RunnableConfig):
        required_fields = cls.model_json_schema()['required']
        cfg = {f: runnable["configurable"][f] for f in required_fields}
        configurable = cls(**cfg)
        return configurable


@dataclass(kw_only=True)
class ConfigurationBase:
    """
        * Modified from: https://github.com/langchain-ai/research-rabbit/blob/main/src/research_rabbit/configuration.py
    """

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "ConfigurationBase":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})



class GraphBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_response(self, input_dict: dict[str, Any], verbose: bool = False):
        pass

    @abstractmethod
    def build_graph(self):
        pass


class SearchQuery(BaseModel):
    """
    A data model representing a single web search query.
    
    This class encapsulates a search query string that can be used with web search
    APIs or services. It serves as a structured way to represent and validate
    search queries within the AI common framework.
    
    Attributes:
        search_query (str): The actual search query string to be executed.
                           Can be None if not yet defined.
    
    Example:
        >>> query = SearchQuery(search_query="artificial intelligence trends 2024")
        >>> print(query.search_query)
        "artificial intelligence trends 2024"
        
        >>> empty_query = SearchQuery()
        >>> print(empty_query.search_query)
        None
    """
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    """
    A collection model for managing multiple web search queries.
    
    This class provides a structured container for handling multiple SearchQuery
    instances. It's designed to facilitate batch processing of search queries
    and maintain consistency across search operations in AI workflows.
    
    The class validates that all contained queries are properly structured
    SearchQuery instances and provides a clean interface for accessing and
    manipulating the query collection.
    
    Attributes:
        queries (List[SearchQuery]): A list of SearchQuery instances representing
                                   individual search queries to be processed.
                                   Each query in the list must be a valid
                                   SearchQuery object.
    
    Example:
        >>> query1 = SearchQuery(search_query="machine learning algorithms")
        >>> query2 = SearchQuery(search_query="deep learning frameworks")
        >>> queries_collection = Queries(queries=[query1, query2])
        >>> print(len(queries_collection.queries))
        2
        >>> print(queries_collection.queries[0].search_query)
        "machine learning algorithms"
        
        >>> # Empty collection
        >>> empty_queries = Queries(queries=[])
        >>> print(len(empty_queries.queries))
        0
        
        >>> # Adding queries programmatically
        >>> new_queries = Queries(queries=[])
        >>> new_queries.queries.append(SearchQuery(search_query="AI ethics"))
        >>> print(len(new_queries.queries))
        1
    
    Note:
        This class is typically used in conjunction with web search nodes
        and other components that need to process multiple search queries
        in a structured manner.
    """
    queries: List[SearchQuery] = Field(description="List of search queries.")