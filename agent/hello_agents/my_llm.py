import os
from typing import Optional

from hello_agents import HelloAgentsLLM
from openai import OpenAI


class  MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        if provider == "modelscope":
            print("Using modelscope as the provider.")
            self.provider = "modelscope"

            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or os.getenv("MODELSCOPE_BASE_URL")
            self.model = model or os.getenv("MODELSCOPE_MODEL_ID")
            if not self.api_key or not self.base_url or not self.model:
                raise ValueError("Modelscope requires api_key, base_url, and model.")
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            super().__init__(model, api_key, base_url, provider, **kwargs)
