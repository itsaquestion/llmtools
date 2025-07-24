"""
llmtools - A Python library for parallel LLM inference processing with optional SQLite database storage and recovery capabilities.

This library provides tools for efficiently processing multiple LLM prompts in parallel
while optionally storing all prompts and results in a SQLite database for persistence
and analysis. It includes robust recovery functionality to resume interrupted processing
sessions and recover incomplete results.

Key Components:
- ParallelLLMProcessor: Main class for parallel LLM processing with database storage and recovery
- DatabaseManager: Thread-safe SQLite database manager for storing results
- DatabaseValidator: Utility class for validating database files and table structures
- RecoveryAnalyzer: Analyzer for identifying incomplete results and extracting recovery data
- RecoveryProcessor: Processor for reprocessing failed prompts during recovery operations
- DatabaseUpdater: Utility class for updating database records with recovery results
- OrderedResult: Data class for maintaining result order in concurrent processing

Features:
- Parallel processing with configurable worker threads
- Optional real-time SQLite database storage
- Recovery from interrupted processing sessions
- Thread-safe database operations
- Robust error handling and retry mechanisms
- Context manager support for automatic cleanup

Examples:
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
    
    Recovery from interrupted session:
    >>> processor = ParallelLLMProcessor(chat_fn=my_llm_function, num_workers=4)
    >>> results = processor.recover_from_database("interrupted_session.db")
    >>> processor.close()
"""

from .openrouter import chat, print_to_screen, save_to_file, dummy_callback
from .parallel_llm_processor import ParallelLLMProcessor, OrderedResult
from .database_manager import DatabaseManager
from .database_validator import DatabaseValidator
from .recovery_analyzer import RecoveryAnalyzer
from .recovery_processor import RecoveryProcessor
from .database_updater import DatabaseUpdater
from .utils import write_text, read_text

# Version information
__version__ = "1.0.0"
__author__ = "llmtools contributors"

# Main exports
__all__ = [
    # Core processing classes
    "ParallelLLMProcessor",
    "DatabaseManager", 
    "DatabaseValidator",
    "RecoveryAnalyzer",
    "RecoveryProcessor",
    "DatabaseUpdater",
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