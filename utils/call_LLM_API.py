""" 
    该库提供同步和异步调用LLM的接口,仅支持openAI接口的api调用,如果需要调用其他LLM接口,请自行修改代码
    API和base_url通过环境变量DASHSCOPE_API_KEY和DASHSCOPE_API_BASE传入,如果不传入,则默认使用openAI的API和base_url
"""

import os
import threading
from openai import OpenAI,AsyncOpenAI

class call_LLM_API:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://api.openai.com/v1")
        self.total_tokens_used = 0
        self._token_lock = threading.Lock()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _extract_total_tokens(self, response):
        if getattr(response, "usage", None) is None:
            return 0

        total_tokens = getattr(response.usage, "total_tokens", None)
        if total_tokens is not None:
            return int(total_tokens)

        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        return prompt_tokens + completion_tokens

    def _add_tokens(self, tokens):
        if tokens <= 0:
            return
        with self._token_lock:
            self.total_tokens_used += tokens

    def call(self, prompt, model="qwen3.5-plus-2026-02-15", timeout=60, enable_thinking=False, count_tokens=True):
        """
        同步调用LLM接口,返回json对象

        :param prompt: 输入的文本提示
        :param model: 使用的模型,默认为qwen3.5-plus-2026-02-15
        :param timeout: 超时时间,默认为60秒
        :param enable_thinking: 是否启用思考过程,默认为False
        :param count_tokens: 是否统计本次调用token数,默认为True

        
        :return: LLM返回的json对象
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                extra_body={
                    "enable_thinking": enable_thinking,
                },
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout  # 设置超时时间，避免无限等待
            )

            if count_tokens:
                self._add_tokens(self._extract_total_tokens(response))

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"调用LLM接口失败: {str(e)}")

    def get_total_tokens_used(self):
        """
        获取当前实例累计已使用的token数量

        :return: 累计token数
        """
        with self._token_lock:
            return self.total_tokens_used
    
    def reset_total_tokens_used(self):
        """
        重置累计token数量为0
        """        
        with self._token_lock:
            self.total_tokens_used = 0

    async def call_async(self, prompt, model="qwen3.5-plus-2026-02-15", timeout=60, enable_thinking=False, count_tokens=True):
        """
        异步调用LLM接口,返回json对象

        :param prompt: 输入的文本提示
        :param model: 使用的模型,默认为qwen3.5-plus-2026-02-15
        :param timeout: 超时时间,默认为60秒
        :param enable_thinking: 是否启用思考过程,默认为False
        :param count_tokens: 是否统计本次调用token数,默认为True

        :return: LLM返回的json对象
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                extra_body={
                    "enable_thinking": enable_thinking,
                },
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout  # 设置超时时间，避免无限等待
            )

            if count_tokens:
                self._add_tokens(self._extract_total_tokens(response))

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"调用LLM接口失败: {str(e)}")
