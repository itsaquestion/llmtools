"""
Recovery processor module for reprocessing failed prompts.

This module provides the RecoveryProcessor class which is responsible for:
- Reprocessing failed prompts using the existing parallel processing configuration
- Integrating with the existing retry mechanisms and concurrent processing
- Providing progress display during reprocessing
- Handling errors and logging during the recovery process
"""

import logging
from typing import List, Tuple, Dict, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Type checking import to avoid circular imports
if TYPE_CHECKING:
    from .parallel_llm_processor import ParallelLLMProcessor

# Configure logging
logger = logging.getLogger(__name__)


class RecoveryProcessor:
    """
    Processor for reprocessing failed prompts during recovery operations.
    
    This class integrates with the existing ParallelLLMProcessor to reprocess
    failed prompts using the same configuration (chat_fn, num_workers, retry settings).
    It provides progress tracking and detailed error handling for the recovery process.
    """
    
    def __init__(self, parallel_processor: 'ParallelLLMProcessor'):
        """
        Initialize the RecoveryProcessor with a reference to the main processor.
        
        Args:
            parallel_processor: The ParallelLLMProcessor instance to use for reprocessing
        """
        self.processor = parallel_processor
        self.chat_fn = parallel_processor.chat_fn
        self.num_workers = parallel_processor.num_workers
        self.retry_attempts = parallel_processor.retry_attempts
        self.retry_delay = parallel_processor.retry_delay
        self.timeout = parallel_processor.timeout
        
        logger.debug(f"RecoveryProcessor initialized with {self.num_workers} workers, "
                    f"{self.retry_attempts} retry attempts, {self.timeout}s timeout")
    
    def process_failed_prompts(self, failed_records: List[Tuple[int, str]]) -> Dict[int, str]:
        """
        Reprocess failed prompts and return a mapping of record IDs to new results.
        
        This method:
        1. Takes a list of failed records (id, prompt) tuples
        2. Reprocesses each prompt using the existing parallel processing configuration
        3. Displays progress during processing
        4. Handles errors gracefully and logs detailed information
        5. Returns a dictionary mapping record IDs to their new results
        
        Args:
            failed_records: List of (record_id, prompt) tuples to reprocess
            
        Returns:
            Dict[int, str]: Mapping of record IDs to their new results
            
        Raises:
            RuntimeError: If critical errors occur during processing
        """
        if not failed_records:
            logger.info("No failed records to process")
            return {}
        
        # Enhanced logging: Processing start with detailed configuration
        logger.info(f"Initializing reprocessing of {len(failed_records)} failed records")
        logger.info(f"Reprocessing configuration:")
        logger.info(f"  - Parallel workers: {self.num_workers}")
        logger.info(f"  - Retry attempts per record: {self.retry_attempts}")
        logger.info(f"  - Retry delay: {self.retry_delay}s (with exponential backoff)")
        logger.info(f"  - Timeout per attempt: {self.timeout}s")
        
        # Dictionary to store results mapped by record ID
        results_map = {}
        
        # Enhanced tracking statistics
        successful_count = 0
        failed_count = 0
        total_attempts = 0
        total_processing_time = 0
        
        try:
            import time
            processing_start_time = time.time()
            
            # Use ThreadPoolExecutor with the same configuration as the main processor
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(self._process_single_failed_record, record_id, prompt): record_id
                    for record_id, prompt in failed_records
                }
                
                # Enhanced progress bar with more detailed description and performance metrics
                progress_desc = f"Reprocessing {len(failed_records)} failed prompts"
                with tqdm(total=len(failed_records), desc=progress_desc, 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                         smoothing=0.1) as pbar:
                    
                    completed_count = 0
                    for future in as_completed(future_to_record):
                        record_id = future_to_record[future]
                        completed_count += 1
                        
                        try:
                            result = future.result()
                            results_map[record_id] = result
                            successful_count += 1
                            
                            # Enhanced progress logging
                            logger.debug(f"✓ Record {record_id} reprocessed successfully ({completed_count}/{len(failed_records)})")
                            
                            # Update progress bar description with current stats
                            success_rate = (successful_count / completed_count * 100)
                            pbar.set_description(f"Reprocessing ({success_rate:.0f}% success rate)")
                            
                        except Exception as e:
                            # Store error message as result
                            error_msg = f"Reprocessing failed: {str(e)}"
                            results_map[record_id] = error_msg
                            failed_count += 1
                            
                            # Enhanced error logging
                            logger.error(f"✗ Record {record_id} reprocessing failed ({completed_count}/{len(failed_records)}): {str(e)}")
                            
                            # Update progress bar description with current stats
                            success_rate = (successful_count / completed_count * 100) if completed_count > 0 else 0
                            pbar.set_description(f"Reprocessing ({success_rate:.0f}% success rate)")
                        
                        pbar.update(1)
                        
                        # Log milestone progress
                        if completed_count % max(1, len(failed_records) // 10) == 0 or completed_count == len(failed_records):
                            current_success_rate = (successful_count / completed_count * 100) if completed_count > 0 else 0
                            logger.info(f"Progress: {completed_count}/{len(failed_records)} completed "
                                       f"({current_success_rate:.1f}% success rate)")
            
            total_processing_time = time.time() - processing_start_time
            
            # Enhanced processing summary with detailed statistics
            logger.info("Reprocessing phase completed")
            logger.info(f"Processing statistics:")
            logger.info(f"  - Total records processed: {len(failed_records)}")
            logger.info(f"  - Successful reprocessing: {successful_count}")
            logger.info(f"  - Failed reprocessing: {failed_count}")
            logger.info(f"  - Success rate: {(successful_count / len(failed_records) * 100):.1f}%")
            logger.info(f"  - Total processing time: {total_processing_time:.2f} seconds")
            logger.info(f"  - Average time per record: {(total_processing_time / len(failed_records)):.2f} seconds")
            
            if failed_count > 0:
                logger.warning(f"⚠️  {failed_count} records failed during reprocessing and will contain error messages")
                logger.warning("These records may need manual review or different processing parameters")
            
            if successful_count > 0:
                logger.info(f"✅ {successful_count} records successfully reprocessed and ready for database update")
            
            return results_map
            
        except Exception as e:
            logger.error(f"Critical error during reprocessing: {str(e)}")
            logger.error(f"Processing was interrupted after {successful_count + failed_count} records")
            raise RuntimeError(f"Reprocessing operation failed: {str(e)}")
    
    def _process_single_failed_record(self, record_id: int, prompt: str) -> str:
        """
        Process a single failed record with retry logic.
        
        This method replicates the retry and timeout logic from the main processor
        but is specifically designed for recovery operations.
        
        Args:
            record_id: The database record ID
            prompt: The prompt to reprocess
            
        Returns:
            str: The result from reprocessing the prompt
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempt = 0
        last_exception = None
        
        logger.debug(f"Starting reprocessing of record {record_id}: {prompt[:50]}...")
        
        # Handle zero retry attempts case
        if self.retry_attempts == 0:
            error_msg = f"Failed after 0 attempts: No retry attempts configured"
            logger.error(f"Record {record_id} failed permanently: {error_msg}")
            raise Exception(error_msg)
        
        while attempt < self.retry_attempts:
            try:
                # Use a separate ThreadPoolExecutor for timeout handling
                with ThreadPoolExecutor(max_workers=1) as timeout_executor:
                    future = timeout_executor.submit(self.chat_fn, prompt)
                    result = future.result(timeout=self.timeout)
                
                logger.debug(f"Successfully reprocessed record {record_id} on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                attempt += 1
                last_exception = e
                
                logger.debug(f"Attempt {attempt} failed for record {record_id}: {str(e)}")
                
                if attempt < self.retry_attempts:
                    # Calculate exponential backoff delay
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.debug(f"Retrying record {record_id} in {delay:.2f}s (attempt {attempt + 1}/{self.retry_attempts})")
                    time.sleep(delay)
                else:
                    # All attempts failed
                    error_msg = f"Failed after {self.retry_attempts} attempts: {str(last_exception)}"
                    logger.error(f"Record {record_id} failed permanently: {error_msg}")
                    raise Exception(error_msg)
        
        # This should never be reached, but included for completeness
        raise Exception(f"Unexpected error: retry loop completed without result for record {record_id}")
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get current processing configuration statistics.
        
        Returns:
            Dict[str, any]: Dictionary containing processing configuration details
        """
        return {
            'num_workers': self.num_workers,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'chat_function': str(self.chat_fn.__name__) if hasattr(self.chat_fn, '__name__') else str(type(self.chat_fn))
        }
    
    def validate_configuration(self) -> bool:
        """
        Validate that the processor configuration is suitable for recovery operations.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check if chat function is callable
            if not callable(self.chat_fn):
                logger.error("Chat function is not callable")
                return False
            
            # Check numeric parameters
            if self.num_workers < 1:
                logger.error(f"Invalid num_workers: {self.num_workers}")
                return False
            
            if self.retry_attempts < 0:
                logger.error(f"Invalid retry_attempts: {self.retry_attempts}")
                return False
            
            if self.retry_delay < 0:
                logger.error(f"Invalid retry_delay: {self.retry_delay}")
                return False
            
            if self.timeout <= 0:
                logger.error(f"Invalid timeout: {self.timeout}")
                return False
            
            logger.debug("Recovery processor configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False