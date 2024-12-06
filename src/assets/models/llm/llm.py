import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self, model = 'gemini-1.5-pro-latest'):
        genai.configure(api_key=os.getenv('GENAI_API_KEY'))
        self.model = genai.GenerativeModel(model)

    def __call__(self, text: str):
        prompt = f"Generate a simple sentence using the following keywords: {text}"
        try:
            res = self.model.generate_content(prompt).text
        except Exception as e:
            # change api key
            res = ""

        return res