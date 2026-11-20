"""Filesystem-only setup check, collected and run by Pytest."""

from pathlib import Path


def test_essential_project_structure_exists() -> None:
    """Verify the scaffold without importing ML packages or using services."""
    project_root = Path(__file__).resolve().parent.parent
    required_files = (
        "distribuai/__init__.py",
        "distribuai/config.py",
        "distribuai/ingestion/__init__.py",
        "distribuai/rag/__init__.py",
        "distribuai/ui/__init__.py",
        "data/uploads/.gitkeep",
        "data/vector_store/.gitkeep",
        "tests/test_setup.py",
        "requirements.txt",
        ".gitignore",
        "README.md",
    )
    required_directories = (
        "distribuai",
        "distribuai/ingestion",
        "distribuai/rag",
        "distribuai/ui",
        "data/uploads",
        "data/vector_store",
        "tests",
    )
    for relative_path in required_files:
        assert (project_root / relative_path).is_file(), relative_path
    for relative_path in required_directories:
        assert (project_root / relative_path).is_dir(), relative_path
