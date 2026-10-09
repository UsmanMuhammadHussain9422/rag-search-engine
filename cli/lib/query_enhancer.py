from typing import Optional
from openai.types.chat import ChatCompletionMessageParam
from llm.openrouter import openrouter

class Query_Enhancer:
    def __init__(self) -> None:
        self.spell_system_prompt: ChatCompletionMessageParam = {
            "role": "system",
            "content": """Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        """
        }
        self.spell_user_prompt: ChatCompletionMessageParam = {
            "role": "user",
            "content": ""
        }

    def spell_correct(self, query: str) -> str:
        self.spell_user_prompt["content"] = query
        corrected = ""
        
        OpenRouter = openrouter()        
        response = OpenRouter.trigger_prompt([self.spell_system_prompt, self.spell_user_prompt])
        if response is not None:
            corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query


    def enhance_query(self, query: str, method: Optional[str] = None) -> str:
        match method:
            case "spell":
                return self.spell_correct(query)
            case _:
                return query
