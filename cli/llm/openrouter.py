import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

# logging.basicConfig(level=logging.DEBUG)  # shows requests/urllib3 debug info

class openrouter():
    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        
        self.model: str = os.environ.get("LLM_MODEL") or "openrouter/free"

        self.llm_type = os.environ.get("LLM")
        if not self.model:
            raise RuntimeError("LLM environment variable not set")

        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def trigger_prompt(self, prompts: list[ChatCompletionMessageParam], verbose = False) -> ChatCompletion | None:
        if not prompts or len(prompts) == 0:
            if verbose:
                print("Prompt Required for a Response.")
                print(f"Prompt tokens: 0")
                print(f"Response tokens: 0")
            return None
        
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,  # Check OpenRouter's catalog for current IDs
            messages=prompts
        )

        if verbose:
            print(response.choices[0].message.content)
            if response.usage is not None:
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Response tokens: {response.usage.completion_tokens}")
            else:
                print("Token usage unavailable.")

        return response