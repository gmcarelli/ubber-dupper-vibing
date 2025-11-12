import os
from pathlib import Path

from llm_host.chat_session import ChatSession

def process_files(
    folder_path: str,
    chat_session: ChatSession,
    guidelines_file: str = "fleubers.txt",
):
    """
    Processes files in a folder using a ChatSession to interact with an LLM.

    Args:
        folder_path (str): The path to the folder containing the files.
        chat_session (ChatSession): The chat session to use for LLM interaction.
        guidelines_file (str, optional): The name of the guidelines file. Defaults to "fleubers.txt".
    """
    project_root = Path(os.getcwd())
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    folder_name = os.path.basename(folder_path)
    output_file = results_dir / f"{folder_name}.txt"

    guidelines_path = Path(folder_path) / guidelines_file
    if not guidelines_path.is_file():
        raise FileNotFoundError(f"Guidelines file '{guidelines_file}' not found in '{folder_path}'")

    guidelines_content = guidelines_path.read_text(encoding="utf-8")

    # The first message sent to the chat session will be stored as history
    chat_session.send_message(guidelines_content)

    for filename in os.listdir(folder_path):
        if filename == guidelines_file:
            continue

        file_path = Path(folder_path) / filename
        if file_path.is_file():
            file_content = file_path.read_text(encoding="utf-8")

            # Send the file content to the LLM
            response = chat_session.send_message(file_content)

            # Save the response
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(response + "\n")
