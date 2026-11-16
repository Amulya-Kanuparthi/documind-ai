from pathlib import Path
import pytest
from documind.core import load_documents, make_chunks, split_text

def test_overlap():
    result = split_text("one two three four five six", size=4, overlap=2)
    assert result == ["one two three four", "three four five six", "five six"]

def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        split_text("hello", size=3, overlap=3)

def test_sample_policy_loads():
    records = load_documents(Path("data/sample"))
    assert any("24 hours" in text for _, _, text in records)
    assert make_chunks(records)[0].source == "vendor_policy.txt"
