
import random


from openai import OpenAI


from clivis import preference

import tiktoken

TOTAL_TOKENS = 0  # Global token counter

__all__ = [
    "basic_llm_chat",
    "basic_single_step_llm_chat",
    "count_tokens",
]

def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

qwen_client = OpenAI(
    api_key=preference.QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

deepseek_client = OpenAI(api_key=preference.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")






def basic_llm_chat(messages,
                   model=preference.LLM_MODEL_NAME,
                   temperature=0.7,
                   top_p=0.95,
                   max_tokens=4096,
                   stop=None,
                   frequency_penalty=0.0,
                   presence_penalty=1.5):
    """
    Basic chat function using the OpenAI API.
    :param messages: Message list
    :param model: Model name
    :param temperature: Temperature parameter
    :param top_p: Top-p sampling parameter
    :param max_tokens: Maximum token count
    :param stop: Stop sequence
    :param frequency_penalty: Frequency penalty
    :param presence_penalty: Presence penalty
    :param n: Number of results to return
    :param stream: Whether to stream results
    :return: Generated message content
    """
    global TOTAL_TOKENS
    try:
        response = qwen_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False,
            seed=random.randint(0, 100000)
        )

        # response = deepseek_client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=messages,
        #     temperature=temperature,
        #     top_p=top_p,
        #     max_tokens=max_tokens,
        #     stop=stop,
        #     frequency_penalty=frequency_penalty,
        #     presence_penalty=presence_penalty,
        #     stream=False,
        #     seed=random.randint(0, 100000)
        # )
    except Exception as e:
        print(f"Chat Error: {e}")
        return None

    result = response.choices[0].message.content if response.choices else None
    token_count = count_tokens(str(result))
    TOTAL_TOKENS += token_count
    return result


def basic_single_step_llm_chat(input_text,
                model=preference.LLM_MODEL_NAME,
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
                stop=None,
                frequency_penalty=0.0,
                presence_penalty=0.0):
    """
    Basic single-step chat function using the OpenAI API.
    :param input_text: Input text
    :param model: Model name
    :param temperature: Temperature parameter
    :param top_p: Top-p sampling parameter
    :param max_tokens: Maximum token count
    :param stop: Stop sequence
    :param frequency_penalty: Frequency penalty
    :param presence_penalty: Presence penalty
    :return: Generated message content
    """
    try:
        response = qwen_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            seed=random.randint(0, 100000)
        )
    except Exception as e:
        print(f"Chat Error: {e}")
        return None

    return response.choices[0].message.content if response.choices else None

