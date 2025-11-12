import pytest
from unittest.mock import MagicMock
from llm_host.ollama_client import OllamaClient, OllamaConnectionError

@pytest.fixture
def mock_ollama_client_backend():
    """Fixture to create a mock of the internal ollama.Client."""
    return MagicMock()

@pytest.fixture
def ollama_client(mocker, mock_ollama_client_backend):
    """Fixture to create an OllamaClient instance with a mocked backend."""
    mocker.patch('ollama.Client', return_value=mock_ollama_client_backend)
    client = OllamaClient()
    client.connect("http://localhost:11434")
    return client

def test_connect_raises_error_on_failure(mocker):
    """Tests that connect raises OllamaConnectionError on connection failure."""
    mocker.patch('ollama.Client', side_effect=Exception("Connection failed"))
    client = OllamaClient()
    with pytest.raises(OllamaConnectionError):
        client.connect("http://localhost:11434")

def test_chat_success(ollama_client, mock_ollama_client_backend):
    """Tests a successful chat call."""
    expected_response = "This is a test response."
    mock_ollama_client_backend.chat.return_value = {
        "message": {"content": expected_response}
    }

    messages = [{"role": "user", "content": "Hello"}]
    response = ollama_client.chat("test_model", messages)

    assert response == expected_response
    mock_ollama_client_backend.chat.assert_called_once()
