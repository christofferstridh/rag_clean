from pathlib import Path


def test_project_structure_exists() -> None:
    assert Path("src/main").exists()
