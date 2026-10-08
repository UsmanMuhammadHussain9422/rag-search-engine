import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
# logging.basicConfig(level=logging.DEBUG)  # shows requests/urllib3 debug info

class openrouter():
    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        
        self.model = os.environ.get("LLM_MODEL")
        if not self.model:
            self.model = "openrouter/free"

        self.llm_type = os.environ.get("LLM")
        if not self.model:
            raise RuntimeError("LLM environment variable not set")

        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def trigger_prompt(self):
        response = self.client.chat.completions.create(
            model=self.model,  # Check OpenRouter's catalog for current IDs
            messages=[
                    {
                    "role": "user",
                    "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
                }
            ],
        )

        print(response.choices[0].message.content)
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Response tokens: {response.usage.completion_tokens}")
