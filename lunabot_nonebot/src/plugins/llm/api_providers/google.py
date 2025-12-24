from ...llm.api_provider import *
from google import genai
from google.genai.types import (
    Content, 
    Part, 
    GenerateContentConfig, 
    GenerateContentResponse, 
    HttpOptions,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    ThinkingConfig,
)
import asyncio
import json
import os
from io import BytesIO


class GenaiCompletions:
    def __init__(self, client: genai.Client):
        self.genai_client = client
        safety_categories = [
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        ]
        self.safety_settings = [SafetySetting(
            category=category,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ) for category in safety_categories]
    
    async def create(
        self,
        model: str, 
        messages: list, 
        extra_body: dict = None, 
        max_tokens: int = None,
        thinking_config: dict = {}
    ):
        image_response = extra_body.get("image_response", False)

        # 转换内容格式
        system_prompt = None
        contents: List[Content] = []
        for item in messages:
            role, content = item['role'], item['content']
            if role == "system_prompt":
                assert isinstance(content, str)
                system_prompt = content
            else:
                role = "model" if role == "assistant" else "user"
                parts: List[Part] = []
                if isinstance(content, str):
                    parts.append(Part.from_text(text=content))
                else:
                    for part in content:
                        if part['type'] == "text":
                            parts.append(Part.from_text(text=part['text']))
                        elif part['type'] == "image_url":
                            image_base64: str = part['image_url']['url']
                            assert image_base64.startswith("data:image/jpeg;base64,")
                            image_bytes = base64.b64decode(image_base64.split(",")[1])
                            parts.append(Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
                if parts:
                    contents.append(Content(role=role, parts=parts))

        # 生成回复
        config = GenerateContentConfig(
            response_modalities=['Text', 'Image'] if image_response else ['Text'],
            system_instruction=system_prompt,
            safety_settings=self.safety_settings,
            max_output_tokens=max_tokens,
        )
        config.thinking_config = ThinkingConfig(**thinking_config)

        # thinking_config = {}
        # if include_thoughts is not None: 
        #     thinking_config['include_thoughts'] = include_thoughts
        # if include_thoughts != False and 'lite' not in model:
        #     if '3' in model:
        #         thinking_config['thinking_level'] = 'high' if high_thinking_level else 'low'
        #     else:
        #         thinking_config['thinking_budget'] = -1 if high_thinking_level else '128'

        def gen():
            return self.genai_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        resp: GenerateContentResponse = await run_in_pool(gen)

        candidate = resp.candidates[0]
        if not candidate.content:
            raise Exception(f"生成失败，可能是由于生成图片被屏蔽，请更换提示词")

        prompt_tokens = resp.usage_metadata.prompt_token_count
        completion_tokens = resp.usage_metadata.candidates_token_count

        # 转换回复格式
        result = { 
            "choices": [{
                "message": {
                    "content": [],
                    "reasoning_content": "",
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens if prompt_tokens else 0,
                "completion_tokens": completion_tokens if completion_tokens else 0,
            }
        }

        if candidate.content.parts is None:
            raise Exception("返回内容为空")

        for part in candidate.content.parts:
            if not part.thought:
                if part.text is not None:
                    result["choices"][0]["message"]['content'].append(part.text)
                elif part.inline_data is not None:
                    img = Image.open(BytesIO(part.inline_data.data))
                    result["choices"][0]["message"]['content'].append(img)
            else:
                if part.text is not None:
                    result["choices"][0]["message"]['reasoning_content'] += part.text
        return result

class GenaiChat:
    def __init__(self, client: genai.Client):
        self.completions = GenaiCompletions(client)

class GenaiEmbeddings:
    def __init__(self, client: genai.Client):
        self.genai_client = client

    async def create(self, input: List[str], model: str, encoding_format: str = 'float'):
        def gen():
            return self.genai_client.models.embed_content(
                model=model,
                contents=input,
            )
        resp = await run_in_pool(gen)
        return resp.embeddings


class GenaiAsyncClient:
    def __init__(self, http_options: dict, api_key: str):
        self.genai_client = genai.Client(
            api_key=api_key, 
            http_options=HttpOptions(**http_options),
        )
        self.chat = GenaiChat(self.genai_client)
        self.embeddings = GenaiEmbeddings(self.genai_client)


class GoogleApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="google", code="gg")

    def get_client(self) -> GenaiAsyncClient:
        return GenaiAsyncClient(
            http_options=self.config.get('http_options'),
            api_key=self.get_api_key(),
        )
    
    async def sync_quota(self):
        return None



