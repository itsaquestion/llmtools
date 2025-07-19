from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time
import logging
from src.llmtools.database_manager import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class OrderedResult:
    index: int
    result: str

class ParallelLLMProcessor:
    """
    A parallel processor for LLM inference with optional SQLite database storage.
    
    This class provides concurrent processing of multiple prompts while maintaining
    result order and optionally storing all prompts and results in a SQLite database
    for persistence and analysis.
    
    Key Features:
    - Concurrent processing with configurable worker threads
    - Automatic retry mechanism with exponential backoff
    - Optional real-time database storage of prompts and results
    - Thread-safe database operations
    - Graceful error handling and resource management
    - Context manager support for automatic cleanup
    
    Database Storage:
    When save_to_db=True, the processor creates a SQLite database with the following structure:
    - Table: llm_results
    - Columns: id (PRIMARY KEY), prompt (TEXT), result (TEXT), created_at (TIMESTAMP)
    - Real-time updates: Results are stored immediately upon completion
    - Thread-safe: Multiple workers can safely update the database concurrently
    
    Example:
        Basic usage without database:
        >>> processor = ParallelLLMProcessor(chat_fn=my_llm_function, num_workers=4)
        >>> results = processor.process_prompts(["Hello", "World"])
        >>> processor.close()
        
        With database storage:
        >>> with ParallelLLMProcessor(
        ...     chat_fn=my_llm_function,
        ...     num_workers=4,
        ...     save_to_db=True,
        ...     db_filename="my_results.db"
        ... ) as processor:
        ...     results = processor.process_prompts(["Hello", "World"])
        
        Using default database filename:
        >>> processor = ParallelLLMProcessor(
        ...     chat_fn=my_llm_function,
        ...     save_to_db=True  # Uses llm_results_{YYYYMMDD_HHMMSS}.db
        ... )
    """
    
    def __init__(self, chat_fn: Callable[[str], str], num_workers: int = 4, 
                 retry_attempts: int = 3, retry_delay: float = 1.0,
                 timeout: float = 60.0, save_to_db: bool = False,
                 db_filename: Optional[str] = None):
        """
        Initialize the parallel processor with optional database storage.
        
        Args:
            chat_fn: The LLM chat function that takes a prompt and returns a string.
                    Should be thread-safe or handle concurrent calls appropriately.
            num_workers: Number of parallel worker threads (default: 4).
                        Higher values increase concurrency but may hit API rate limits.
            retry_attempts: Number of retry attempts for failed requests (default: 3).
                           Set to 0 to disable retries.
            retry_delay: Delay between retries in seconds (default: 1.0).
                        Uses exponential backoff for multiple retries.
            timeout: Timeout in seconds for each individual request (default: 60.0).
                    Prevents hanging on slow API calls.
            save_to_db: Whether to save prompts and results to SQLite database (default: False).
                       When True, enables real-time database storage of all processing data.
            db_filename: Database filename for SQLite storage (optional).
                        If None and save_to_db=True, generates default filename with timestamp.
                        Format: "llm_results_{YYYYMMDD_HHMMSS}.db"
        
        Raises:
            TypeError: If chat_fn is not callable or parameter types are incorrect
            ValueError: If numeric parameters are out of valid ranges
            
        Note:
            Database initialization is lazy - the database file is only created when
            process_prompts() is called with save_to_db=True.
        """
        # Validate parameters
        self._validate_parameters(chat_fn, num_workers, retry_attempts, retry_delay, timeout, save_to_db, db_filename)
        
        self.chat_fn = chat_fn
        self.num_workers = num_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.save_to_db = save_to_db
        
        # Initialize database manager configuration if needed
        self.db_manager: Optional[DatabaseManager] = None
        if self.save_to_db:
            # Generate default filename if not provided
            if db_filename is None:
                db_filename = self._generate_default_filename()
            self.db_filename = db_filename
            # Database manager will be initialized lazily when first needed

    def _validate_parameters(self, chat_fn: Callable[[str], str], num_workers: int, 
                           retry_attempts: int, retry_delay: float, timeout: float,
                           save_to_db: bool, db_filename: Optional[str]) -> None:
        """
        Validate initialization parameters.
        
        Args:
            chat_fn: The LLM chat function
            num_workers: Number of parallel workers
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
            timeout: Timeout for each request
            save_to_db: Whether to save to database
            db_filename: Database filename
            
        Raises:
            ValueError: If any parameter is invalid
            TypeError: If parameter types are incorrect
        """
        if not callable(chat_fn):
            raise TypeError("chat_fn must be callable")
        
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")
        
        if not isinstance(retry_attempts, int) or retry_attempts < 0:
            raise ValueError("retry_attempts must be a non-negative integer")
        
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError("retry_delay must be a non-negative number")
        
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        
        if not isinstance(save_to_db, bool):
            raise TypeError("save_to_db must be a boolean")
        
        if db_filename is not None and not isinstance(db_filename, str):
            raise TypeError("db_filename must be a string or None")
        
        if db_filename is not None and not db_filename.strip():
            raise ValueError("db_filename cannot be empty or whitespace")

    def _generate_default_filename(self) -> str:
        """
        Generate a default database filename with human-readable timestamp.
        
        Returns:
            str: Default filename in format "llm_results_{YYYYMMDD_HHMMSS}.db"
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_results_{timestamp}.db"
        logger.info(f"Generated default database filename: {filename}")
        return filename
    
    def _initialize_database_manager(self) -> bool:
        """
        Initialize the database manager lazily when first needed.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.db_manager is not None:
            return True
        
        try:
            self.db_manager = DatabaseManager(self.db_filename)
            logger.info(f"Database manager initialized with file: {self.db_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {str(e)}")
            logger.warning("Continuing without database functionality")
            self.save_to_db = False
            self.db_manager = None
            return False

    def _process_single_prompt(self, args: tuple[int, str]) -> OrderedResult:
        """
        Process a single prompt with retry and timeout logic.
        
        Args:
            args: Tuple of (index, prompt)
            
        Returns:
            OrderedResult containing the original index and result
        """
        idx, prompt = args
        attempt = 0
        result = None
        
        while attempt < self.retry_attempts:
            try:
                future = ThreadPoolExecutor(max_workers=1).submit(self.chat_fn, prompt)
                result = future.result(timeout=self.timeout)
                
                # Update database with successful result
                if self.save_to_db and self.db_manager:
                    try:
                        self.db_manager.update_result(idx, result)
                        logger.debug(f"Database updated for prompt {idx}")
                    except Exception as db_e:
                        logger.error(f"Database update failed for prompt {idx}: {str(db_e)}")
                        # Continue processing - database errors should not interrupt main flow
                
                return OrderedResult(idx, result)
            except Exception as e:
                attempt += 1
                if attempt == self.retry_attempts:
                    error_msg = f"Failed after {self.retry_attempts} attempts: {str(e)}"
                    
                    # Update database with error result
                    if self.save_to_db and self.db_manager:
                        try:
                            self.db_manager.update_result(idx, f"Error: {error_msg}")
                            logger.debug(f"Database updated with error for prompt {idx}")
                        except Exception as db_e:
                            logger.error(f"Database error update failed for prompt {idx}: {str(db_e)}")
                            # Continue processing - database errors should not interrupt main flow
                    
                    raise TimeoutError(error_msg)
                time.sleep(self.retry_delay)

    def process_prompts(self, prompts: List[str]) -> List[str]:
        """
        Process multiple prompts in parallel while maintaining order.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            List of results in the same order as input prompts
        """
        if not prompts:
            logger.warning("Empty prompts list provided")
            return []
        
        # Initialize database records before processing if database is enabled and prompts exist
        if self.save_to_db and prompts:
            # Initialize database manager lazily
            if self._initialize_database_manager():
                try:
                    logger.info(f"Initializing database with {len(prompts)} prompts")
                    if not self.db_manager.initialize_database(prompts):
                        logger.error("Database initialization failed, continuing without database functionality")
                        self.save_to_db = False
                    else:
                        logger.info("Database initialization completed successfully")
                except Exception as e:
                    logger.error(f"Database initialization error: {str(e)}")
                    logger.warning("Continuing without database functionality")
                    self.save_to_db = False
        
        results = [None] * len(prompts)
        tasks = list(enumerate(prompts))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_single_prompt, task): task[0]
                for task in tasks
            }
            
            with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
                for future in as_completed(futures):
                    try:
                        ordered_result = future.result()
                        results[ordered_result.index] = ordered_result.result
                    except Exception as e:
                        idx = futures[future]
                        error_msg = f"Error: {str(e)}"
                        results[idx] = error_msg
                        
                        # Update database with error if not already updated in _process_single_prompt
                        if self.save_to_db and self.db_manager:
                            try:
                                self.db_manager.update_result(idx, error_msg)
                                logger.debug(f"Database updated with error for prompt {idx}")
                            except Exception as db_e:
                                logger.error(f"Database error update failed for prompt {idx}: {str(db_e)}")
                                # Continue processing - database errors should not interrupt main flow
                    
                    pbar.update(1)
        
        return results

    def close(self) -> None:
        """
        Close database connections and clean up resources.
        
        This method should be called when the processor is no longer needed
        to ensure proper cleanup of database resources.
        """
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                self.db_manager.close_connection()
                logger.info("Database manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing database manager: {str(e)}")

    def __enter__(self):
        """
        Context manager entry point.
        
        Returns:
            ParallelLLMProcessor: Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Ensures proper resource cleanup.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred  
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            None: Does not suppress exceptions
        """
        self.close()
        return None

    def __del__(self):
        """
        Destructor to ensure proper cleanup of resources.
        """
        try:
            self.close()
        except Exception:
            # Ignore exceptions in destructor to avoid issues during cleanup
            pass

# Example usage:
if __name__ == "__main__":
    import time
    
    def mock_chat_fn(prompt: str) -> str:
        """Mock LLM function for testing"""
        time.sleep(0.5)  # Simulate API delay
        return f"Response to: {prompt}"

    # Test prompts
    test_prompts = [
        "What is 8 squared?",
        "What color is a ripe banana?",
        "What is the chemical symbol for gold?",
        "How many legs does a spider have?",
        "What is the capital city of Japan?"
    ]
    
    # Initialize and run processor
    processor = ParallelLLMProcessor(chat_fn=mock_chat_fn, num_workers=3,save_to_db=True)
    results = processor.process_prompts(test_prompts)