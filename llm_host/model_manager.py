from .protocols import LLMTools

class ModelManager:
    """Manages LLM models by interacting with a toolset that conforms to LLMTools."""

    def __init__(self, llm_tools: LLMTools):
        self.llm_tools: LLMTools = llm_tools

    def list_models(self) -> str:
        """Lists all available models using the provided LLM toolset."""
        return self.llm_tools.list_models()

    def create_model(self, base_model: str, model_name: str, system_role: str) -> None:
        """Creates a new model using the provided LLM toolset."""
        self.llm_tools.create_model(base_model, model_name, system_role)
