from typing import List, Dict, Any
from .protocols import LLMClientInterface

class ChatSession:
    """Manages a chat session with an LLM, maintaining a single-turn history."""

    def __init__(self, llm_tools: LLMClientInterface, model_name: str):
        self.llm_tools: LLMClientInterface = llm_tools
        self.model_name: str = model_name
        self._history: List[Dict[str, Any]] = []

    def send_message(self, message: str) -> str:
        """
        Sends a message to the LLM and returns the response.
        Maintains a history with only the first user message as context.
        """
        if not self._history:
            # First message from the user, store it as context
            self._history.append({"role": "user", "content": message})

        # Subsequent calls will use the first message as history
        current_messages: List[Dict[str, Any]] = self._history
        if len(self._history) > 0 and self._history[0]['content'] != message:
            # If the current message is different from the stored history,
            # append it for the current call. This handles the case where the
            # first message is the guideline and subsequent messages are user prompts.
            current_messages = self._history + [{"role": "user", "content": message}]

        response_content: str = self.llm_tools.chat(self.model_name, current_messages)
        return response_content
