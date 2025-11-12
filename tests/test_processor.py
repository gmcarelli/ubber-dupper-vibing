import pytest
from unittest.mock import MagicMock
from pathlib import Path
import os
from llm_host.chat_session import ChatSession
from improved_goggles.processor import process_files

@pytest.fixture
def mock_chat_session() -> MagicMock:
    """Fixture to create a mock of the ChatSession."""
    session = MagicMock(spec=ChatSession)
    session.send_message.return_value = "Mocked LLM response."
    return session

@pytest.fixture
def temp_dir_for_processor(tmp_path):
    """Creates a temporary directory with test files for the processor."""
    data_dir = tmp_path / "processor_test_data"
    data_dir.mkdir()
    (data_dir / "fleubers.txt").write_text("Test guidelines.")
    (data_dir / "file1.txt").write_text("Content of file 1.")
    return data_dir

def test_process_files_happy_path(temp_dir_for_processor, mock_chat_session):
    """Tests the successful execution of the process_files function."""
    process_files(str(temp_dir_for_processor), mock_chat_session)

    # Verify that the session was used correctly
    # Called once for guidelines, once for file1.txt
    assert mock_chat_session.send_message.call_count == 2

    # Verify the output file
    project_root = Path(os.getcwd())
    output_file = project_root / "results" / f"{temp_dir_for_processor.name}.txt"
    assert output_file.exists()

    content = output_file.read_text()
    # The mocked response should be in the file
    assert "Mocked LLM response." in content

    os.remove(output_file)
