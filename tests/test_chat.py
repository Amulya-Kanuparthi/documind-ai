
import sys
from pathlib import Path
from unittest.mock import patch

from documind.chat import main


def test_chat_answers_question(tmp_path, capsys):
    """CLI displays answers and retrieved sources."""

    fake_result = {
        "question": "When must accounts be disabled?",
        "answer": "Within 24 hours. [1]",
        "sources": [
            {
                "source": "access_control.txt",
                "page": 1,
                "chunk_id": 0,
            }
        ],
    }

    with (
        patch.object(
            sys,
            "argv",
            [
                "documind.chat",
                "--index",
                str(tmp_path),
                "--method",
                "hybrid",
            ],
        ),
        patch(
            "builtins.input",
            side_effect=[
                "When must accounts be disabled?",
                "exit",
            ],
        ),
        patch(
            "documind.chat.ask_documind",
            return_value=fake_result,
        ) as mock_ask,
    ):
        main()

    output = capsys.readouterr().out

    assert "Within 24 hours. [1]" in output
    assert "access_control.txt" in output
    assert "Goodbye!" in output
    mock_ask.assert_called_once()


def test_chat_exit(tmp_path, capsys):
    """CLI exits without calling the LLM."""

    with (
        patch.object(
            sys,
            "argv",
            ["documind.chat", "--index", str(tmp_path)],
        ),
        patch("builtins.input", return_value="exit"),
        patch("documind.chat.ask_documind") as mock_ask,
    ):
        main()

    assert "Goodbye!" in capsys.readouterr().out
    mock_ask.assert_not_called()


def test_chat_empty_question(tmp_path):
    """Blank questions do not invoke the RAG pipeline."""

    with (
        patch.object(
            sys,
            "argv",
            ["documind.chat", "--index", str(tmp_path)],
        ),
        patch(
            "builtins.input",
            side_effect=["   ", "quit"],
        ),
        patch("documind.chat.ask_documind") as mock_ask,
    ):
        main()

    mock_ask.assert_not_called()


def test_chat_missing_index(tmp_path):
    """CLI rejects a nonexistent index directory."""

    import pytest

    missing = tmp_path / "missing"

    with patch.object(
        sys,
        "argv",
        ["documind.chat", "--index", str(missing)],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
