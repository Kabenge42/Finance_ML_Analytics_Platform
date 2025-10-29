#!/usr/bin/env python3


class NotebookCellFactory:
    """Factory class for creating Jupyter notebook cells with consistent structure."""

    # Constants for cell structure
    NEWLINE_SUFFIX = "\n"

    @staticmethod
    def create_markdown_cell(content: str) -> dict:
        """Create a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [content + NotebookCellFactory.NEWLINE_SUFFIX],
        }

    @staticmethod
    def create_code_cell(code: str) -> dict:
        """Create a code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code + NotebookCellFactory.NEWLINE_SUFFIX],
        }

    @staticmethod
    def create_phase_header(phase_number: str, title: str, description: list[str]) -> dict:
        """Create a standardized phase header markdown cell."""
        content = f"## Phase {phase_number} — {title}\n\n"
        content += "\n".join(f"{i}. {desc}" for i, desc in enumerate(description, 1))
        return NotebookCellFactory.create_markdown_cell(content)

    @staticmethod
    def create_section_header(phase: str, section: str, title: str) -> dict:
        """Create a standardized section header markdown cell."""
        content = f"### {phase}.{section} — {title}"
        return NotebookCellFactory.create_markdown_cell(content)
