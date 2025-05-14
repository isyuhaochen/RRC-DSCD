import os
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API configuration
ARK_API_KEY = ""
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class ApiProxy:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def call(self, model_name, system_prompt, user_prompt, max_tokens=64, headers=None):
        if headers is None:
            headers = {}

        headers['Content-Type'] = 'application/json'
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": [
                # {"role": "system", "content": system_prompt},  # Uncomment if system prompt is needed
                {"role": "user", "content": user_prompt},
            ]
        }

        endpoint = f"{self.base_url}/chat/completions"
        attempt = 0
        max_backoff_time = 32

        while True:
            try:
                response = self.session.post(endpoint, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                return response.json()
            except Exception as e:
                attempt += 1
                wait_time = min(2 ** attempt, max_backoff_time)
                print(f"Attempt {attempt} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)


def get_model_response(prompt, model="deepseek-v3-241226"):
    proxy = ApiProxy(api_key=ARK_API_KEY, base_url=BASE_URL)
    response = proxy.call(model_name=model, system_prompt="", user_prompt=prompt, max_tokens=8196)

    if isinstance(response, str):
        return response

    return response['choices'][0]['message']['content'].strip()


def get_model_response_with_reasoning(prompt, model="deepseek-r1-250120"):
    proxy = ApiProxy(api_key=ARK_API_KEY, base_url=BASE_URL)
    response = proxy.call(model_name=model, system_prompt="", user_prompt=prompt, max_tokens=8196)

    if isinstance(response, str):
        return response

    message = response['choices'][0]['message']
    content = message['content'].strip()
    reasoning = message.get('reasoning_content', '').strip()

    return content, reasoning
