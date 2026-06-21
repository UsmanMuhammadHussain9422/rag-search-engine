import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

class Query_Enhancer:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model= os.environ.get("GEMINI_MODEL")
        self.spell_prompt = """Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """
        
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)

    def spell_correct(self, query: str) -> str:
        if not self.model:
            raise RuntimeError("GEMINI_MODEL environment variable not set")
        
        prompt = self.spell_prompt.format(query=query)
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        corrected = (response.text or "").strip().strip('"')
        return corrected if corrected else query


    def enhance_query(self, query: str, method: Optional[str] = None) -> str:
        match method:
            case "spell":
                return self.spell_correct(query)
            case _:
                return query
