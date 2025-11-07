import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Adiciona o diretório raiz do projeto ao sys.path
# Isso permite que o módulo llm-host seja encontrado
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from llm_host.ollama_client import OllamaClient


def process_files(
    folder_path: str,
    model_name: str,
    client: OllamaClient,
    guidelines_file: str = "fleubers.txt",
):
    """
    Processa os arquivos em uma pasta usando um LLM.

    Args:
        folder_path (str): O caminho para a pasta contendo os arquivos.
        model_name (str): O nome do modelo Ollama a ser usado.
        client (OllamaClient): O cliente Ollama a ser usado.
        guidelines_file (str, optional): O nome do arquivo de diretrizes. Defaults to "fleubers.txt".
    """
    # Garante que a pasta de resultados exista
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Define o arquivo de saída
    folder_name = os.path.basename(folder_path)
    output_file = results_dir / f"{folder_name}.txt"

    # Lê o arquivo de diretrizes
    guidelines_path = Path(folder_path) / guidelines_file
    if not guidelines_path.is_file():
        raise FileNotFoundError(f"Arquivo de diretrizes '{guidelines_file}' não encontrado em '{folder_path}'")

    guidelines_content = guidelines_path.read_text(encoding="utf-8")

    # Prepara o histórico de mensagens inicial com as diretrizes
    # Usaremos o papel 'system' para as diretrizes, que é uma prática comum
    base_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": guidelines_content}
    ]

    # Itera sobre os arquivos na pasta
    for filename in os.listdir(folder_path):
        if filename == guidelines_file:
            continue

        file_path = Path(folder_path) / filename
        if file_path.is_file():
            file_content = file_path.read_text(encoding="utf-8")

            # Adiciona o conteúdo do arquivo como uma mensagem do usuário
            messages = base_messages + [{"role": "user", "content": file_content}]

            # Chama o LLM
            response = client.chat(model_name, messages)

            # Salva a resposta
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(response + "\n")
