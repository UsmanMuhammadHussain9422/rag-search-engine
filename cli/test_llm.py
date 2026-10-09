from llm.openrouter import openrouter
from openai.types.chat import ChatCompletionMessageParam

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]


OpenRouter = openrouter()
OpenRouter.trigger_prompt(messages, True)