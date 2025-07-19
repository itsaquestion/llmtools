import sqlite3
import threading
import logging
import time
from typing import List, Optional
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Thread-safe SQLite database manager for storing LLM processing results.
    
    This class handles database connections, table creation, and thread-safe
    operations for storing and updating LLM inference results.
    """
    
    def __init__(self, db_filename: str):
        """
        Initialize the DatabaseManager with a database file.
        
        Args:
            db_filename: Path to the SQLite database file
        """
        self.db_filename = db_filename
        self._connection = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # Initialize database connection
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """
        Initialize the database connection with thread safety enabled.
        
        Creates a new SQLite connection with check_same_thread=False to allow
        usage across multiple threads.
        """
        try:
            with self._lock:
                self._connection = sqlite3.connect(
                    self.db_filename, 
                    check_same_thread=False,
                    timeout=30.0  # 30 second timeout for database operations
                )
                # Enable WAL mode for better concurrent access
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.commit()
                logger.info(f"Database connection initialized: {self.db_filename}")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for thread-safe database connection access.
        
        Yields:
            sqlite3.Connection: The database connection object
        """
        with self._lock:
            if self._connection is None:
                raise RuntimeError("Database connection not initialized")
            yield self._connection
    
    def create_table(self) -> bool:
        """
        Create the llm_results table if it doesn't exist.
        
        Creates a table with the following structure:
        - id: INTEGER PRIMARY KEY (corresponds to prompt list index + 1)
        - prompt: TEXT NOT NULL (input prompt)
        - result: TEXT (LLM inference result, initially NULL)
        - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP (creation time)
        
        Returns:
            bool: True if table creation was successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if table already exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='llm_results'
                """)
                
                if cursor.fetchone():
                    logger.info("Table 'llm_results' already exists, reusing existing structure")
                    return True
                
                # Create the table
                cursor.execute("""
                    CREATE TABLE llm_results (
                        id INTEGER PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        result TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Table 'llm_results' created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}")
            return False
    
    def initialize_database(self, prompts: List[str]) -> bool:
        """
        Initialize the database by creating the table and batch inserting prompt records.
        
        This method:
        1. Creates the llm_results table if it doesn't exist
        2. Batch inserts all prompts with id corresponding to their index + 1
        3. Sets result field to NULL initially
        4. Ensures database record order matches input prompt list order
        
        Args:
            prompts: List of prompt strings to initialize in the database
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # First create the table
            if not self.create_table():
                logger.error("Failed to create table during database initialization")
                return False
            
            # Check if table already has data
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM llm_results")
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    logger.warning(f"Table already contains {existing_count} records. Skipping initialization.")
                    return True
            
            # Batch insert all prompts
            if prompts:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Prepare data for batch insert
                    # id = index + 1, prompt = prompt text, result = NULL (default)
                    data = [(i + 1, prompt) for i, prompt in enumerate(prompts)]
                    
                    cursor.executemany("""
                        INSERT INTO llm_results (id, prompt) 
                        VALUES (?, ?)
                    """, data)
                    
                    conn.commit()
                    logger.info(f"Successfully initialized database with {len(prompts)} prompt records")
                    
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            return False
    
    def update_result(self, prompt_index: int, result: str, max_retries: int = 3, retry_delay: float = 0.1) -> bool:
        """
        Update the result field for a specific prompt record in a thread-safe manner.
        
        This method updates the result field of the record corresponding to the given
        prompt index. It includes retry mechanism for handling temporary database locks
        or connection issues.
        
        Args:
            prompt_index: The original index of the prompt in the input list (0-based)
            result: The LLM inference result or error message to store
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retry attempts in seconds (default: 0.1)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self._initialized:
            logger.warning("Database not initialized, skipping result update")
            return False
        
        # Convert 0-based index to 1-based database id
        db_id = prompt_index + 1
        
        for attempt in range(max_retries + 1):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Update the result field for the specific record
                    cursor.execute("""
                        UPDATE llm_results 
                        SET result = ? 
                        WHERE id = ?
                    """, (result, db_id))
                    
                    # Check if the update affected any rows
                    if cursor.rowcount == 0:
                        logger.warning(f"No record found with id {db_id} for prompt index {prompt_index}")
                        return False
                    
                    conn.commit()
                    logger.debug(f"Successfully updated result for prompt index {prompt_index} (db_id: {db_id})")
                    return True
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries:
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Database operational error updating prompt {prompt_index}: {str(e)}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to update result for prompt index {prompt_index}: {str(e)}")
                if attempt < max_retries:
                    logger.info(f"Retrying update operation (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                return False
        
        logger.error(f"Failed to update result for prompt index {prompt_index} after {max_retries + 1} attempts")
        return False
    
    def close_connection(self) -> None:
        """
        Close the database connection safely.
        
        This method should be called when the DatabaseManager is no longer needed
        to ensure proper cleanup of database resources.
        """
        try:
            with self._lock:
                if self._connection:
                    self._connection.close()
                    self._connection = None
                    logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    
    def __enter__(self):
        """
        Context manager entry point.
        
        Returns:
            DatabaseManager: Self for use in with statement
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
        self.close_connection()
        return None
    
    def __del__(self):
        """
        Destructor to ensure database connection is closed.
        """
        self.close_connection()