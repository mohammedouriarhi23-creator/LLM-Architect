import logging
from typing import Optional
import httpx
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)


class LMStudioClient:
    """
    Client for interacting with LM Studio's local OpenAI-compatible server
    (default: http://localhost:1234/v1).
    Used for:
      1. Synthetic dataset generation to prepare ShareGPT/Alpaca JSONL for Unsloth.
      2. Running inference with the fine-tuned LLMarchitect GGUF model.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or settings.lmstudio_base_url).rstrip("/")
        self.api_key = api_key or settings.lmstudio_api_key
        self.model = model or settings.lmstudio_model
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=25.0,
            max_retries=0,
        )

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as http:
                resp = await http.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_loaded_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=1.5) as http:
                resp = await http.get(f"{self.base_url}/models")
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    return [
                        {"id": m.get("id", "local-model"), "owned_by": m.get("owned_by", "lm-studio")}
                        for m in data
                    ]
            return []
        except Exception:
            return []

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> Optional[str]:
        if not await self.is_available():
            return None
        try:
            loaded = await self.list_loaded_models()
            active_model = loaded[0]["id"] if loaded else self.model
            response = await self.client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.info("LM Studio server not reachable (%s). Using built-in fallback.", exc)
            return None
