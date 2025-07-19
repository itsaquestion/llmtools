"""
llmtools - A Python library for parallel LLM inference processing with optional SQLite database storage.

This library provides tools for efficiently processing multiple LLM prompts in parallel
while optionally storing all prompts and results in a SQLite database for persistence
and analysis.

Key Components:
- ParallelLLMProcessor: Main class for parallel LLM processing with database storage
- DatabaseManager: Thread-safe SQLite database manager for storing results
- OrderedResult: Data class for maintaining result order in concurrent processing

Example:
    Basic usage without database:
    >>> from llmtools import ParallelLLMProcessor
    >>> processor = ParallelLLMProcessor(chat_fn=my_llm_function, num_workers=4)
    >>> results = processor.process_prompts(["Hello", "World"])
    >>> processor.close()
    
    With database storage:
    >>> with ParallelLLMProcessor(
    ...     chat_fn=my_llm_function,
    ...     save_to_db=True,
    ...     db_filename="results.db"
    ... ) as processor:
    ...     results = processor.process_prompts(["Hello", "World"])
"""

from .openrouter import chat, print_to_screen, save_to_file, dummy_callback
from .parallel_llm_processor import ParallelLLMProcessor, OrderedResult
from .database_manager import DatabaseManager
from .utils import write_text, read_text

# Version information
__version__ = "1.0.0"
__author__ = "llmtools contributors"

# Main exports
__all__ = [
    # Core processing classes
    "ParallelLLMProcessor",
    "DatabaseManager", 
    "OrderedResult",
    
    # OpenRouter integration
    "chat",
    "print_to_screen", 
    "save_to_file",
    "dummy_callback",
    
    # Utility functions
    "write_text",
    "read_text",
    
    # Package metadata
    "__version__",
    "__author__"
]