import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Optional, TypeAlias, Literal

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


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
    A comprehensive data model representing a structured web search query.
    
    This class encapsulates all components of a search query including the actual
    query string, the specific aspect being searched, and the rationale behind
    the query generation. It serves as a structured way to represent, validate,
    and document search queries within the AI common framework.
    
    The SearchQuery model is designed to provide context and traceability for
    search operations, making it easier to understand why specific queries were
    generated and what information they aim to retrieve.
    
    Attributes:
        search_query (str): The actual search query string to be executed
                           against web search APIs or services.
        aspect (str, optional): Describes which specific aspect or facet of the topic
                              the query aims to cover (e.g., "recent developments",
                              "technical specifications", "market trends"). Defaults to None.
        rationale (str, optional): The reasoning or justification for generating this
                                 particular search query, explaining why it was chosen
                                 and what information it's expected to retrieve. Defaults to None.

    """
    search_query: str = Field(description="Query for web search.")
    aspect: str = Field(None, description="Which aspect of the topic the query aims to cover.")
    rationale: str = Field(None, description="Reasoning for generating the search query.")


class Queries(BaseModel):
    queries: list[SearchQuery] = Field(description="List of search queries.")
