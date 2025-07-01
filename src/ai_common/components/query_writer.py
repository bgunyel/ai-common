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
        """
        Generate targeted web search queries using LLM with structured JSON output.
        
        This method orchestrates the generation of a specified number of diverse, 
        targeted search queries designed to gather comprehensive information about 
        a given research topic. It leverages a language model with structured JSON 
        output to ensure consistent query formatting and comprehensive topic coverage.
        
        The method performs the following operations:
        1. Validates that the input state contains a required 'topic' attribute
        2. Extracts configuration parameters from the runnable config
        3. Formats detailed instructions including the topic, current date, and query count
        4. Invokes the LLM with JSON response format to generate structured queries
        5. Tracks token usage for cost monitoring and performance analysis
        6. Parses the JSON response and converts it to SearchQuery objects
        7. Updates the state with generated queries and execution metadata
        
        The generated queries are designed to:
        - Cover different aspects and facets of the research topic
        - Account for the most current information available as of today's date
        - Be specific enough to avoid generic or irrelevant search results
        - Be diverse enough to gather comprehensive information for summary writing
        
        Args:
            state (BaseModel): The current workflow state containing the research topic.
                             Must have a 'topic' attribute (str) that defines the subject
                             for which search queries will be generated.
            config (RunnableConfig): The runnable configuration containing execution
                                   parameters. Expected to contain configurable settings
                                   including 'number_of_queries' that specifies how many
                                   search queries to generate.
        
        Returns:
            BaseModel: The updated state object with the following modifications:
                      - search_queries: List of SearchQuery objects, each containing
                        search_query, aspect, and rationale fields
                      - steps: Updated list with NodeBase.QUERY_WRITER appended to track
                        the execution flow
                      - token_usage: Updated dictionary tracking input and output tokens
                        consumed by the LLM, organized by model name
        
        Raises:
            AttributeError: If the input state does not have a required 'topic' attribute
            json.JSONDecodeError: If the LLM response cannot be parsed as valid JSON
            KeyError: If the parsed JSON does not contain the expected 'queries' field
            
        Example:
            >>> state = ResearchState(topic="artificial intelligence in healthcare")
            >>> config = RunnableConfig(configurable={"number_of_queries": 3})
            >>> updated_state = query_writer.run(state, config)
            >>> len(updated_state.search_queries)
            3
            >>> updated_state.search_queries[0].search_query
            "AI applications in medical diagnosis 2024"
        
        Note:
            - Uses structured JSON output format to ensure consistent query generation
            - Incorporates today's date in instructions to bias toward current information
            - Tracks detailed token usage for cost monitoring and performance optimization
            - All generated queries follow the SearchQuery schema with search_query,
              aspect, and rationale fields
        """
        if not hasattr(state, 'topic'):
            raise AttributeError("State must have a 'topic' attribute")
        
        configurable = get_config_from_runnable(
            configuration_module_prefix = self.configuration_module_prefix,
            config = config
        )
        state.steps.append(NodeBase.QUERY_WRITER)
        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format = {"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
            json_dict = json.loads(results.content)
        state.search_queries = [SearchQuery(**q) for q in json_dict['queries']]
        return state
