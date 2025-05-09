"""Utility module for file operations.

This module provides basic functions for reading from and writing to text files.
"""
def read_text(file: str) -> str:
    """Read text from a file.
    
    Args:
        file: Path to the file to read
        
    Returns:
        The text content of the file
    """
    with open(file, 'r', encoding='utf8') as f:
        text = f.read()
        return text


def write_text(file: str, text: str) -> None:
    """Write text to a file.
    
    Args:
        file: Path to the file to write
        text: Text content to write to the file
    """
    with open(file, 'w', encoding='utf8') as f:
        f.write(text)