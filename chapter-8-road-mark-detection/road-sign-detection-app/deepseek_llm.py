import requests
from pandasai.llm.base import LLM

class DeepSeekLLM(LLM):
    def __init__(self, api_key: str, model: str = "deepseek-flash"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    def call(self, instruction, context: dict = None, **kwargs) -> str:
        if not isinstance(instruction, str):
            instruction = str(instruction)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant for data analysis."},
                {"role": "user", "content": instruction}
            ],
            "temperature": 0.0
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise Exception(f"DeepSeek API Error: {resp.status_code}, {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    @property
    def type(self):
        return "deepseek-custom"