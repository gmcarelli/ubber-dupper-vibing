from dataclasses import dataclass
import ollama
from typing import List, Dict, Any
from logger_config import log


@dataclass
class ParameterOptions:
    temperature: float | None
    top_p: float | None


class OllamaConnectionError(Exception):
    pass


class OllamaClient:

    _client: ollama.Client
    options: ParameterOptions = {"temperature": 0.0, "top_p": 0.9}

    def connect_to_host(self, host_url: str) -> "OllamaClient":
        try:
            self._client = ollama.Client(host=host_url)
            log.info(f"Cliente Ollama conectado com sucesso ao host: {host_url}")
            return self
        except Exception as e:
            error_message = f"Não foi possível conectar ao servidor Ollama em {host_url}. Detalhes: {e}"
            log.error(error_message)
            raise OllamaConnectionError(error_message) from e

    def create_model(self, base_model: str, model_name: str, system_role: str) -> None:
        try:
            # modelfile = f"FROM {base_model}\nSYSTEM {system_role}"
            self._client.create(model=model_name, from_=base_model, system=system_role)
            log.info(
                f"Modelo personalizado '{model_name}' criado com sucesso a partir do modelo base '{base_model}'."
            )
        except Exception as e:
            log.error(f"Erro ao criar o modelo personalizado '{model_name}': {e}")

    def list_models(self) -> str:
        try:
            response = self._client.list()
            if response:
                formatted_output: str = "Resposta completa da API:\n"

                # Exibir modelos de forma mais legível
                for model in response["models"]:
                    # Acessando os atributos corretamente do objeto Model
                    model_name = model.model
                    model_size = model.size

                    # Formatando o tamanho para uma representação mais legível
                    size_gb = model_size / (1024**3)  # Converter bytes para GB

                    formatted_output += f"{model_name}\n"
                    formatted_output += f"Tamanho: {size_gb:.2f} GB\n"
                    formatted_output += "\n"

            return formatted_output

        except Exception as e:
            log.error(f"Erro ao listar os modelos: {e}")
            return ""

    def chat(self, model_name: str, message: str) -> str:
        try:
            log.info(f"Enviando prompt para o modelo '{model_name}'...")
            response = self._client.chat(
                model=model_name,
                messages=[{"role": "user", "content": message}],
                options={
                    "temperature": self.options.temperature,
                    "top_p": self.options.top_p,
                },
            )
            content = response.get("message", {}).get("content", "")
            log.info("Resposta recebida do modelo.")
            return content
        except Exception as e:
            log.error(f"Erro durante o chat com o modelo '{model_name}': {e}")
            return ""


if __name__ == "__main__":
    pass
