from enum import Enum

class LlmServers(Enum): # Alphabetical Order
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    OPENAI = 'openai'
    # Self-serve
    OLLAMA = 'ollama'
    VLLM = 'vllm'

class NodeBase(Enum):
    # In alphabetical order
    QUERY_WRITER = 'query_writer'
    WEB_SEARCH = 'web_search'


def build_node_enum(base_enum: type[Enum], extra_nodes: dict[str, str]) -> type[Enum]:
    """
    Builds a new Enum class called `Node` by extending the provided `base_enum`
    with additional entries from `extra_nodes`. If a key conflict exists, the
    value from `base_enum` takes precedence.

    Args:
        base_enum (Enum): The base enum to extend.
        extra_nodes (dict): Additional enum members to add.

    Returns:
        Enum: A new Enum class named `Node` with combined members.
    """
    base_members = {e.name: e.value for e in base_enum}

    # Resolve conflicts: base_enum values take precedence
    merged_members = {**extra_nodes, **base_members}

    return Enum('Node', merged_members)
