from typing import List, Dict, Any
import ollama
from .protocols import HostConnector, LLMClientInterface
from .logger_config import log

# Constants
TEMPERATURE: float = 0.0
TOP_P: float = 0.9

class OllamaConnectionError(Exception):
    pass

class OllamaClient(HostConnector, LLMClientInterface):
    """Concrete implementation of HostConnector and LLMClientInterface for Ollama."""

    def __init__(self):
        self._client: ollama.Client | None = None

    def connect(self, host_url: str) -> None:
        """Connects to the Ollama host."""
        try:
            self._client = ollama.Client(host=host_url)
            log.info(f"Successfully connected to Ollama host: {host_url}")
        except Exception as e:
            error_message = f"Failed to connect to Ollama server at {host_url}. Details: {e}"
            log.error(error_message)
            raise OllamaConnectionError(error_message) from e

    def list_models(self) -> str:
        """Lists all models available on the Ollama host."""
        if not self._client:
            raise OllamaConnectionError("Client is not connected. Please call connect() first.")

        try:
            response = self._client.list()
            formatted_output: str = "Models:\n"
            for model in response["models"]:
                model_name = model.get('name', 'N/A')
                size = model.get('size', 0)
                size_gb = size / (1024**3)
                formatted_output += f"- {model_name} ({size_gb:.2f} GB)\n"
            return formatted_output
        except Exception as e:
            log.error(f"Error listing models: {e}")
            return ""

    def create_model(self, base_model: str, model_name: str, system_role: str) -> None:
        """Creates a custom model on the Ollama host."""
        if not self._client:
            raise OllamaConnectionError("Client is not connected. Please call connect() first.")

        try:
            self._client.create(model=model_name, from_=base_model, system=system_role)
            log.info(f"Custom model '{model_name}' created successfully from base model '{base_model}'.")
        except Exception as e:
            log.error(f"Error creating custom model '{model_name}': {e}")

    def chat(self, model_name: str, messages: List[Dict[str, Any]]) -> str:
        """Sends a chat request to a model on the Ollama host."""
        if not self._client:
            raise OllamaConnectionError("Client is not connected. Please call connect() first.")

        try:
            log.info(f"Sending prompt to model '{model_name}'...")
            response = self._client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                },
            )
            content: str = response.get("message", {}).get("content", "")
            log.info("Received response from model.")
            return content
        except Exception as e:
            log.error(f"Error during chat with model '{model_name}': {e}")
            return ""
