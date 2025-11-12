from typing import Protocol, List, Dict, Any

class HostConnector(Protocol):
    """Defines the interface for connecting to a host."""
    def connect(self, host_url: str) -> None:
        ...

class LLMTools(Protocol):
    """Defines the interface for interacting with LLM models."""
    def list_models(self) -> str:
        ...

    def create_model(self, base_model: str, model_name: str, system_role: str) -> None:
        ...

    def chat(self, model_name: str, messages: List[Dict[str, Any]]) -> str:
        ...
