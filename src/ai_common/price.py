from typing import Any

from .enums import LlmServers, ModelNames

PRICE_USD_PER_MILLION_TOKENS = {
    LlmServers.GROQ.value: {
        ModelNames.GPT_OSS_120B: {'input_tokens': 0.15, 'output_tokens': 0.75},
        ModelNames.GPT_OSS_20B: {'input_tokens': 0.10, 'output_tokens': 0.50},
    },
    LlmServers.OLLAMA.value: {
        ModelNames.GPT_OSS_120B: {'input_tokens': 0.00, 'output_tokens': 0.00},
        ModelNames.GPT_OSS_20B: {'input_tokens': 0.00, 'output_tokens': 0.00},
    },
    LlmServers.OPENAI.value: {
        'gpt-4.1': {'input_tokens': 2.00, 'output_tokens': 8.00},
        'gpt-4.1-mini': {'input_tokens': 0.40, 'output_tokens': 1.60},
        'gpt-4.1-nano': {'input_tokens': 0.10, 'output_tokens': 0.40},
        'gpt-4o': {'input_tokens': 2.50, 'output_tokens': 10.00},
        'gpt-4o-mini': {'input_tokens': 0.15, 'output_tokens': 0.60},
        'o1': {'input_tokens': 15.00, 'output_tokens': 60.00},
        'o3': {'input_tokens': 10.00, 'output_tokens': 40.00},
        'o1-mini': {'input_tokens': 1.10, 'output_tokens': 4.40},
        'o3-mini': {'input_tokens': 1.10, 'output_tokens': 4.40},
    },
    LlmServers.ANTHROPIC.value: {
        'claude-opus-4-latest': {'input_tokens': 15.00, 'output_tokens': 75.00},
        'claude-sonnet-4-latest': {'input_tokens': 3.00, 'output_tokens': 15.00},
        'claude-3-5-haiku-latest': {'input_tokens': 0.80, 'output_tokens': 4.00},
        'claude-3-7-sonnet-latest': {'input_tokens': 3.00, 'output_tokens': 15.00},
    }
}


def calculate_token_cost_for_one_model(params: dict[str, Any], token_usage: dict[str, Any]) -> dict[str, Any]:
    model_provider = params['model_provider']
    model = params['model']
    price_dict = PRICE_USD_PER_MILLION_TOKENS[model_provider][model]
    cost = sum([price_dict[k] * token_usage[model][k] for k in price_dict.keys()]) / 1e6
    return {
        'model_provider': model_provider,
        'model': model,
        'cost': cost,
    }

def calculate_token_cost(llm_config: dict[str, Any], token_usage: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    total_cost = 0
    cost_list = []
    for model_type, params in llm_config.items():
        cost_dict = calculate_token_cost_for_one_model(params = params, token_usage = token_usage)
        total_cost += cost_dict['cost']
        cost_list.append(cost_dict)
    return cost_list, total_cost
