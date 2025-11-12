import pytest
from unittest.mock import MagicMock
from llm_host.protocols import LLMClientInterface
from llm_host.chat_session import ChatSession

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Fixture to create a mock of the LLMClientInterface."""
    return MagicMock(spec=LLMClientInterface)

def test_chat_session_send_message_first_time(mock_llm_client):
    """Tests the first message in a ChatSession, which should establish history."""
    mock_llm_client.chat.return_value = "Response to guideline."
    session = ChatSession(mock_llm_client, "test_model")

    guideline = "You are a helpful assistant."
    response = session.send_message(guideline)

    assert response == "Response to guideline."
    assert session._history == [{"role": "user", "content": guideline}]
    mock_llm_client.chat.assert_called_once_with("test_model", [{"role": "user", "content": guideline}])

def test_chat_session_send_message_second_time(mock_llm_client):
    """Tests the second message, which should use the established history."""
    mock_llm_client.chat.return_value = "Response to prompt."
    session = ChatSession(mock_llm_client, "test_model")

    guideline = "You are a helpful assistant."
    session.send_message(guideline)

    prompt = "Hello, world!"
    response = session.send_message(prompt)

    assert response == "Response to prompt."
    assert session._history == [{"role": "user", "content": guideline}]

    expected_messages = [
        {"role": "user", "content": guideline},
        {"role": "user", "content": prompt}
    ]
    mock_llm_client.chat.assert_called_with("test_model", expected_messages)
