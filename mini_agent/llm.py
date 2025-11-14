"""LLM client for MiniMax M2 via Anthropic-compatible API."""

import logging
from typing import Any

import httpx

from .retry import RetryConfig as RetryConfigBase
from .retry import async_retry
from .schema import FunctionCall, LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


class LLMClient:
    """MiniMax M2 LLM Client via Anthropic-compatible endpoint.

    Supported models:
    - MiniMax-M2
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimax.io/anthropic",
        model: str = "MiniMax-M2",
        retry_config: RetryConfigBase | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfigBase()

        # Callback for tracking retry count
        self.retry_callback = None

    def _is_lm_studio(self) -> bool:
        """Check if connecting to LM Studio (OpenAI-compatible API)

        Returns:
            True if api_base contains lm studio endpoints
        """
        return "127.0.0.1" in self.api_base or "localhost" in self.api_base

    async def _make_api_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute API request (core method that can be retried)

        Args:
            payload: Request payload

        Returns:
            API response result

        Raises:
            Exception: API call failed
        """
        # Build headers based on API type (Anthropic vs OpenAI/LM Studio)
        if self._is_lm_studio():
            # LM Studio / OpenAI format
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            # Anthropic format
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_base}",
                headers=headers,
                json=payload,
            )

            result = response.json()

        # Check for errors (Anthropic format)
        if result.get("type") == "error":
            error_info = result.get("error", {})
            error_msg = f"API Error ({error_info.get('type')}): {error_info.get('message')}"
            raise Exception(error_msg)

        # Check for MiniMax base_resp errors
        if "base_resp" in result:
            base_resp = result["base_resp"]
            status_code = base_resp.get("status_code")
            status_msg = base_resp.get("status_msg")

            if status_code not in [0, 1000, None]:
                error_msg = f"MiniMax API Error (code {status_code}): {status_msg}"
                if status_code == 1008:
                    error_msg += "\n\n⚠️  Insufficient account balance, please recharge on MiniMax platform"
                elif status_code == 2013:
                    error_msg += f"\n\n⚠️  Model '{self.model}' is not supported"
                raise Exception(error_msg)

        return result

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate response from LLM."""
        # Check if using LM Studio (OpenAI-compatible)
        is_lm_studio = self._is_lm_studio()

        # Extract system message
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            # For user and assistant messages
            if msg.role in ["user", "assistant"]:
                if is_lm_studio:
                    # LM Studio / OpenAI format - simple messages
                    if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                        # For LM Studio, we'll pass thinking as part of content
                        content = msg.content or ""
                        if msg.thinking:
                            # LM Studio doesn't have thinking blocks, append to content
                            content = f"[Thinking: {msg.thinking}]\n{content}"
                        api_messages.append({"role": msg.role, "content": content})
                    else:
                        # Simple text message
                        api_messages.append({"role": msg.role, "content": msg.content})
                else:
                    # Anthropic format - complex content blocks
                    if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                        # Build content blocks for assistant with thinking and/or tool calls
                        content_blocks = []

                        # Add thinking block if present
                        if msg.thinking:
                            content_blocks.append({"type": "thinking", "thinking": msg.thinking})

                        # Add text content if present
                        if msg.content:
                            content_blocks.append({"type": "text", "text": msg.content})

                        # Add tool use blocks
                        if msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                content_blocks.append(
                                    {
                                        "type": "tool_use",
                                        "id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "input": tool_call.function.arguments,
                                    }
                                )

                        api_messages.append({"role": "assistant", "content": content_blocks})
                    else:
                        api_messages.append({"role": msg.role, "content": msg.content})

            # For tool result messages
            elif msg.role == "tool":
                if is_lm_studio:
                    # LM Studio format - simple tool result
                    api_messages.append({"role": "user", "content": msg.content})
                else:
                    # Anthropic format - complex tool_result block
                    api_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.tool_call_id,
                                    "content": msg.content,
                                }
                            ],
                        }
                    )

        # Build request payload
        if is_lm_studio:
            # LM Studio / OpenAI format
            payload = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": 16384,
            }

            # Add system message as part of messages for LM Studio
            if system_message:
                payload["messages"].insert(0, {"role": "system", "content": system_message})

            # Convert tools to OpenAI format if provided
            if tools:
                lm_studio_tools = []
                for tool in tools:
                    # Convert Anthropic format to OpenAI function format
                    lm_studio_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        }
                    })
                payload["tools"] = lm_studio_tools
        else:
            # Anthropic format
            payload = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": 16384,
            }

            # Add system message separately for Anthropic
            if system_message:
                payload["system"] = system_message

            # Add tools if provided (Anthropic format)
            if tools:
                payload["tools"] = tools

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            result = await api_call(payload)
        else:
            # Don't use retry
            result = await self._make_api_request(payload)

        # Parse response based on API type
        if is_lm_studio:
            # LM Studio / OpenAI format - simple response
            try:
                choice = result["choices"][0]
                message = choice.get("message", {})
                text_content = message.get("content", "")
                stop_reason = choice.get("finish_reason", "stop")

                # LM Studio includes reasoning_content directly!
                thinking_content = message.get("reasoning_content", "")

                # Parse tool calls from LM Studio
                tool_calls = []
                for tc in message.get("tool_calls", []):
                    if tc.get("type") == "function":
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id"),
                                type="function",
                                function=FunctionCall(
                                    name=tc.get("function", {}).get("name"),
                                    arguments=tc.get("function", {}).get("arguments", {}),
                                ),
                            )
                        )

            except (KeyError, IndexError) as e:
                raise Exception(f"LM Studio API returned unexpected response format. Response: {result}")
        else:
            # Anthropic format - complex content blocks
            try:
                content_blocks = result.get("content", [])
                stop_reason = result.get("stop_reason", "stop")

                # Extract text content, thinking, and tool calls
                text_content = ""
                thinking_content = ""
                tool_calls = []

                for block in content_blocks:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "thinking":
                        thinking_content += block.get("thinking", "")
                    elif block.get("type") == "tool_use":
                        # Parse Anthropic tool_use block
                        tool_calls.append(
                            ToolCall(
                                id=block.get("id"),
                                type="function",
                                function=FunctionCall(
                                    name=block.get("name"),
                                    arguments=block.get("input", {}),
                                ),
                            )
                        )
            except Exception as e:
                raise Exception(f"Anthropic API returned unexpected response format: {e}")

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=stop_reason,
        )
