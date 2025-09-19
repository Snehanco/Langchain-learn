from typing import Any, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent execution events."""

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Run when LLM starts."""
        print(f"Prompt to LLM was: {prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends."""
        print(f"LLM finished with response: {response.generations[0][0].text}")
        print("********")
