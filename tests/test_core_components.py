import pytest
from unittest.mock import MagicMock
from llm_host.protocols import LLMTools
from llm_host.model_manager import ModelManager
from llm_host.chat_session import ChatSession

@pytest.fixture
def mock_llm_tools() -> MagicMock:
    """Fixture to create a mock of the LLMTools protocol."""
    return MagicMock(spec=LLMTools)

def test_model_manager_list_models(mock_llm_tools):
    """Tests that ModelManager correctly calls list_models on its toolset."""
    mock_llm_tools.list_models.return_value = "llama3\n"
    manager = ModelManager(mock_llm_tools)
    result = manager.list_models()
    assert result == "llama3\n"
    mock_llm_tools.list_models.assert_called_once()

def test_chat_session_send_message_first_time(mock_llm_tools):
    """Tests the first message in a ChatSession, which should establish history."""
    mock_llm_tools.chat.return_value = "Response to guideline."
    session = ChatSession(mock_llm_tools, "test_model")

    guideline = "You are a helpful assistant."
    response = session.send_message(guideline)

    assert response == "Response to guideline."
    # The first message sets the history
    assert session._history == [{"role": "user", "content": guideline}]
    mock_llm_tools.chat.assert_called_once_with("test_model", [{"role": "user", "content": guideline}])

def test_chat_session_send_message_second_time(mock_llm_tools):
    """Tests the second message, which should use the established history."""
    mock_llm_tools.chat.return_value = "Response to prompt."
    session = ChatSession(mock_llm_tools, "test_model")

    # First message (guideline)
    guideline = "You are a helpful assistant."
    session.send_message(guideline)

    # Second message (user prompt)
    prompt = "Hello, world!"
    response = session.send_message(prompt)

    assert response == "Response to prompt."
    # History should remain unchanged
    assert session._history == [{"role": "user", "content": guideline}]

    # Check that chat was called with history + new prompt
    expected_messages = [
        {"role": "user", "content": guideline},
        {"role": "user", "content": prompt}
    ]
    mock_llm_tools.chat.assert_called_with("test_model", expected_messages)
