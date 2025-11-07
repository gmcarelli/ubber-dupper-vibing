import pytest
from unittest.mock import MagicMock
from llm_host.ollama_client import OllamaClient

@pytest.fixture
def mock_ollama_client():
    """Fixture para criar um mock do cliente ollama."""
    return MagicMock()

@pytest.fixture
def ollama_client_instance(mocker, mock_ollama_client):
    """Fixture para criar uma instância de OllamaClient com o cliente ollama mockado."""
    mocker.patch('ollama.Client', return_value=mock_ollama_client)
    client = OllamaClient().connect_to_host("http://localhost:11434")
    return client

def test_chat_success(ollama_client_instance, mock_ollama_client):
    """Testa o método chat em um cenário de sucesso."""
    # Configuração do mock
    expected_response = "Esta é uma resposta de teste."
    mock_ollama_client.chat.return_value = {
        "message": {
            "content": expected_response
        }
    }

    # Chamada do método
    messages = [{"role": "user", "content": "Olá"}]
    response = ollama_client_instance.chat("test_model", messages)

    # Verificações
    assert response == expected_response
    mock_ollama_client.chat.assert_called_once_with(
        model="test_model",
        messages=messages,
        options={
            "temperature": ollama_client_instance.options.temperature,
            "top_p": ollama_client_instance.options.top_p,
        },
    )

def test_chat_api_error(ollama_client_instance, mock_ollama_client):
    """Testa o tratamento de erro quando a API do ollama falha."""
    # Configuração do mock para simular uma exceção
    mock_ollama_client.chat.side_effect = Exception("Falha na API")

    # Chamada do método
    messages = [{"role": "user", "content": "Olá"}]
    response = ollama_client_instance.chat("test_model", messages)

    # Verificações
    assert response == ""
