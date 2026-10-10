from typing import Optional
from openai.types.chat import ChatCompletionMessageParam
from llm.openrouter import openrouter

class Query_Enhancer:
    def __init__(self) -> None:
        self.spell_system_prompt: ChatCompletionMessageParam = {
            "role": "system",
            "content": ""
        }
        self.spell_user_prompt: ChatCompletionMessageParam = {
            "role": "user",
            "content": ""
        }
        self.OpenRouter = openrouter()

    def spell_correct(self, query: str) -> str:
        self.spell_system_prompt["content"] = """Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        """
        self.spell_user_prompt["content"] = query
        corrected = ""
        
        response = self.OpenRouter.trigger_prompt([self.spell_system_prompt, self.spell_user_prompt])
        if response is not None:
            corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query

    def rewrite_query(self, query):
        self.spell_system_prompt["content"] = """Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.
        """
        
        self.spell_user_prompt["content"] = query
        corrected = ""
        
        response = self.OpenRouter.trigger_prompt([self.spell_system_prompt, self.spell_user_prompt])
        if response is not None:
            corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query

    def enhance_query(self, query: str, method: Optional[str] = None) -> str:
        match method:
            case "spell":
                return self.spell_correct(query)
            case "rewrite":
                return self.rewrite_query(query)
            case _:
                return query

