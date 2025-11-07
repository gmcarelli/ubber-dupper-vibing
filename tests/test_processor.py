import pytest
from unittest.mock import MagicMock
from pathlib import Path
import os

from improved_goggles.processor import process_files
from llm_host.ollama_client import OllamaClient

@pytest.fixture
def mock_ollama_client_for_processor():
    """Fixture que cria um mock de OllamaClient para os testes do processador."""
    client = MagicMock(spec=OllamaClient)
    client.chat.return_value = "Resposta mockada do LLM."
    return client

@pytest.fixture
def temp_dir(tmp_path):
    """Cria um diretório temporário com arquivos de teste."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    (data_dir / "fleubers.txt").write_text("Diretrizes de teste.")
    (data_dir / "file1.txt").write_text("Conteúdo do arquivo 1.")
    (data_dir / "file2.txt").write_text("Conteúdo do arquivo 2.")
    return data_dir

def test_process_files_success(temp_dir, mock_ollama_client_for_processor):
    """Testa o cenário de sucesso da função process_files."""
    # Chama a função a ser testada
    process_files(str(temp_dir), "test_model", mock_ollama_client_for_processor)

    # Verifica se o arquivo de resultado foi criado e contém o esperado
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    output_file = results_dir / f"{temp_dir.name}.txt"

    assert output_file.exists()

    content = output_file.read_text()
    # Esperamos duas respostas, uma para cada arquivo (file1.txt e file2.txt)
    expected_calls = 2
    assert content.count("Resposta mockada do LLM.") == expected_calls

    # Verifica as chamadas ao método chat
    assert mock_ollama_client_for_processor.chat.call_count == expected_calls

    # Limpa o arquivo de resultados após o teste
    os.remove(output_file)

def test_process_files_no_guidelines(tmp_path, mock_ollama_client_for_processor):
    """Testa o erro quando o arquivo de diretrizes não é encontrado."""
    data_dir = tmp_path / "no_guidelines_dir"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("some content")

    with pytest.raises(FileNotFoundError):
        process_files(str(data_dir), "test_model", mock_ollama_client_for_processor)
