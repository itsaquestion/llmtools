"""
Database updater module for updating recovery results in the database.

This module provides the DatabaseUpdater class which is responsible for:
- Updating database records with new results from recovery processing
- Using transactions to ensure data integrity
- Implementing batch updates for performance optimization
- Handling concurrent access and database locking issues
"""

import sqlite3
import logging
import time
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseUpdater:
    """
    Utility class for updating database records with recovery results.
    
    This class provides methods for safely updating database records with new results
    while maintaining data integrity and handling concurrent access scenarios.
    """
    
    # Configuration constants
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 0.5  # seconds
    BATCH_SIZE = 500  # Increased batch size for better performance
    LARGE_DATASET_THRESHOLD = 1000  # Threshold for using optimized processing
    
    @staticmethod
    def update_results(db_file_path: str, results_map: Dict[int, str], 
                      timeout: float = DEFAULT_TIMEOUT) -> List[int]:
        """
        Update database records with new results using transactions for data integrity.
        
        This method updates only the records specified in the results_map, preserving
        all other data including original prompts and created_at timestamps.
        
        Args:
            db_file_path: Path to the SQLite database file
            results_map: Dictionary mapping record IDs to new result values
            timeout: Database connection timeout in seconds
            
        Returns:
            List[int]: List of record IDs that failed to update
            
        Raises:
            sqlite3.Error: If database operations fail
            ValueError: If input parameters are invalid
        """
        if not isinstance(results_map, dict):
            raise ValueError("results_map must be a dictionary")
        
        if not results_map:
            logger.info("No results to update")
            return []
        
        # Enhanced logging: Update operation start
        logger.info(f"Starting database update for {len(results_map)} records")
        logger.info(f"Database file: {db_file_path}")
        logger.info(f"Connection timeout: {timeout}s")
        
        failed_updates = []
        
        try:
            import time
            update_start_time = time.time()
            
            # Use optimized processing for large datasets
            if len(results_map) > DatabaseUpdater.LARGE_DATASET_THRESHOLD:
                logger.info(f"Large dataset detected ({len(results_map)} records), using optimized batch processing")
                failed_updates = DatabaseUpdater._update_results_optimized(
                    db_file_path, results_map, timeout
                )
            elif len(results_map) > DatabaseUpdater.BATCH_SIZE:
                logger.info(f"Medium dataset detected ({len(results_map)} records), using standard batch processing")
                failed_updates = DatabaseUpdater._update_results_in_batches(
                    db_file_path, results_map, timeout
                )
            else:
                logger.info(f"Processing {len(results_map)} records in single batch")
                # Single batch update for smaller datasets
                with DatabaseUpdater._get_database_connection(db_file_path, timeout) as conn:
                    failed_updates = DatabaseUpdater._update_batch(conn, results_map)
            
            update_duration = time.time() - update_start_time
            successful_updates = len(results_map) - len(failed_updates)
            
            # Enhanced logging: Update results summary
            logger.info(f"Database update completed in {update_duration:.2f} seconds")
            logger.info(f"Update results summary:")
            logger.info(f"  - Records attempted: {len(results_map)}")
            logger.info(f"  - Successful updates: {successful_updates}")
            logger.info(f"  - Failed updates: {len(failed_updates)}")
            logger.info(f"  - Success rate: {(successful_updates / len(results_map) * 100):.1f}%")
            
            if failed_updates:
                logger.warning(f"Failed to update records: {failed_updates}")
                if len(failed_updates) <= 10:
                    for record_id in failed_updates:
                        logger.warning(f"  - Record {record_id}: Update failed")
                else:
                    logger.warning(f"  - First 5 failed records: {failed_updates[:5]}")
                    logger.warning(f"  - ... and {len(failed_updates) - 5} more failed records")
            else:
                logger.info("✅ All database updates completed successfully")
                
            return failed_updates
            
        except sqlite3.Error as e:
            logger.error(f"Database error during update operation: {str(e)}")
            logger.error(f"Database file: {db_file_path}")
            logger.error(f"Records attempted: {len(results_map)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during update operation: {str(e)}")
            logger.error(f"Database file: {db_file_path}")
            logger.error(f"Records attempted: {len(results_map)}")
            raise RuntimeError(f"Database update failed: {str(e)}")
    
    @staticmethod
    def _update_results_in_batches(db_file_path: str, results_map: Dict[int, str], 
                                  timeout: float) -> List[int]:
        """
        Update results in batches to handle large datasets efficiently.
        
        Args:
            db_file_path: Path to the SQLite database file
            results_map: Dictionary mapping record IDs to new result values
            timeout: Database connection timeout in seconds
            
        Returns:
            List[int]: List of record IDs that failed to update
        """
        failed_updates = []
        items = list(results_map.items())
        total_batches = (len(items) + DatabaseUpdater.BATCH_SIZE - 1) // DatabaseUpdater.BATCH_SIZE
        
        # Enhanced logging: Batch processing start
        logger.info(f"Starting batch processing: {len(items)} updates in {total_batches} batches")
        logger.info(f"Batch size: {DatabaseUpdater.BATCH_SIZE} records per batch")
        
        successful_batches = 0
        failed_batches = 0
        total_successful_updates = 0
        
        # Add progress bar for batch processing
        from tqdm import tqdm
        
        with tqdm(total=total_batches, desc="Database update batches", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for i in range(0, len(items), DatabaseUpdater.BATCH_SIZE):
                batch_items = items[i:i + DatabaseUpdater.BATCH_SIZE]
                batch_map = dict(batch_items)
                batch_num = (i // DatabaseUpdater.BATCH_SIZE) + 1
                
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_map)} records)")
                
                try:
                    import time
                    batch_start_time = time.time()
                    
                    with DatabaseUpdater._get_database_connection(db_file_path, timeout) as conn:
                        batch_failed = DatabaseUpdater._update_batch(conn, batch_map)
                        failed_updates.extend(batch_failed)
                    
                    batch_duration = time.time() - batch_start_time
                    batch_successful = len(batch_map) - len(batch_failed)
                    total_successful_updates += batch_successful
                    successful_batches += 1
                    
                    # Enhanced batch logging
                    logger.debug(f"✓ Batch {batch_num} completed in {batch_duration:.2f}s: "
                               f"{batch_successful}/{len(batch_map)} records updated")
                    
                    # Update progress bar description
                    overall_success_rate = (total_successful_updates / (i + len(batch_map)) * 100)
                    pbar.set_description(f"Database updates ({overall_success_rate:.0f}% success)")
                        
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"✗ Batch {batch_num} failed completely: {str(e)}")
                    # Add all IDs from this batch to failed updates
                    failed_updates.extend(batch_map.keys())
                    
                    # Update progress bar description
                    pbar.set_description(f"Database updates ({failed_batches} batches failed)")
                
                pbar.update(1)
                
                # Log milestone progress for large batch operations
                if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                    processed_records = min(i + DatabaseUpdater.BATCH_SIZE, len(items))
                    current_success_rate = (total_successful_updates / processed_records * 100) if processed_records > 0 else 0
                    logger.info(f"Batch progress: {batch_num}/{total_batches} batches completed "
                               f"({current_success_rate:.1f}% record success rate)")
        
        # Enhanced batch processing summary
        total_failed_updates = len(failed_updates)
        total_attempted_updates = len(items)
        final_success_rate = ((total_attempted_updates - total_failed_updates) / total_attempted_updates * 100) if total_attempted_updates > 0 else 0
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  - Total batches: {total_batches}")
        logger.info(f"  - Successful batches: {successful_batches}")
        logger.info(f"  - Failed batches: {failed_batches}")
        logger.info(f"  - Overall record success rate: {final_success_rate:.1f}%")
        
        if failed_batches > 0:
            logger.warning(f"⚠️  {failed_batches} batches failed completely, affecting {total_failed_updates} records")
        
        return failed_updates
    
    @staticmethod
    def _update_results_optimized(db_file_path: str, results_map: Dict[int, str], 
                                 timeout: float) -> List[int]:
        """
        Optimized update method for very large datasets using prepared statements and connection pooling.
        
        Args:
            db_file_path: Path to the SQLite database file
            results_map: Dictionary mapping record IDs to new result values
            timeout: Database connection timeout in seconds
            
        Returns:
            List[int]: List of record IDs that failed to update
        """
        failed_updates = []
        items = list(results_map.items())
        
        logger.info(f"Starting optimized processing for {len(items)} records")
        
        try:
            with DatabaseUpdater._get_database_connection(db_file_path, timeout) as conn:
                # Additional performance optimizations for large datasets
                conn.execute("PRAGMA synchronous=OFF")  # Faster writes (less safe)
                conn.execute("PRAGMA journal_mode=MEMORY")  # Use memory journal
                conn.execute("PRAGMA cache_size=50000")  # Larger cache
                conn.execute("PRAGMA temp_store=MEMORY")  # Memory temp storage
                
                cursor = conn.cursor()
                
                # Begin single large transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # Prepare statement once for reuse
                update_sql = """
                    UPDATE llm_results 
                    SET result = ? 
                    WHERE id = ? AND (result IS NULL OR result = '' OR result = 'NA')
                """
                
                # Process all updates in single transaction
                successful_updates = 0
                batch_size = 1000  # Larger batch for prepared statements
                
                for i in range(0, len(items), batch_size):
                    batch_items = items[i:i + batch_size]
                    
                    try:
                        # Execute batch of updates
                        for record_id, new_result in batch_items:
                            cursor.execute(update_sql, (new_result, record_id))
                            
                            if cursor.rowcount == 0:
                                failed_updates.append(record_id)
                            else:
                                successful_updates += 1
                        
                        # Log progress for large batches
                        if len(items) > 5000 and (i + batch_size) % 5000 == 0:
                            progress = min(i + batch_size, len(items))
                            logger.info(f"Optimized update progress: {progress}/{len(items)} records processed")
                    
                    except sqlite3.Error as e:
                        logger.error(f"Batch update failed at offset {i}: {str(e)}")
                        # Add all items in this batch to failed updates
                        failed_updates.extend([record_id for record_id, _ in batch_items])
                
                # Commit the transaction
                cursor.execute("COMMIT")
                
                logger.info(f"Optimized update completed: {successful_updates} successful, {len(failed_updates)} failed")
                
        except sqlite3.Error as e:
            logger.error(f"Optimized update transaction failed: {str(e)}")
            # Mark all records as failed
            failed_updates = list(results_map.keys())
        
        return failed_updates
    
    @staticmethod
    def _update_batch(conn: sqlite3.Connection, results_map: Dict[int, str]) -> List[int]:
        """
        Update a batch of records within a single transaction.
        
        Args:
            conn: Database connection
            results_map: Dictionary mapping record IDs to new result values
            
        Returns:
            List[int]: List of record IDs that failed to update
        """
        failed_updates = []
        
        try:
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Prepare update statement
            update_sql = """
                UPDATE llm_results 
                SET result = ? 
                WHERE id = ? AND (result IS NULL OR result = '' OR result = 'NA')
            """
            
            # Process each update
            for record_id, new_result in results_map.items():
                try:
                    cursor.execute(update_sql, (new_result, record_id))
                    
                    # Check if the update actually affected a row
                    if cursor.rowcount == 0:
                        logger.warning(f"Record {record_id} was not updated (may not exist or already complete)")
                        failed_updates.append(record_id)
                    else:
                        logger.debug(f"Successfully updated record {record_id}")
                        
                except sqlite3.Error as e:
                    logger.error(f"Failed to update record {record_id}: {str(e)}")
                    failed_updates.append(record_id)
            
            # Commit transaction
            cursor.execute("COMMIT")
            
        except sqlite3.Error as e:
            # Rollback on any error
            try:
                cursor.execute("ROLLBACK")
            except:
                pass  # Ignore rollback errors
            
            logger.error(f"Transaction failed, rolled back: {str(e)}")
            # Mark all records in this batch as failed
            failed_updates = list(results_map.keys())
            
        return failed_updates
    
    @staticmethod
    @contextmanager
    def _get_database_connection(db_file_path: str, timeout: float):
        """
        Get a database connection with proper configuration and retry logic.
        
        Args:
            db_file_path: Path to the SQLite database file
            timeout: Connection timeout in seconds
            
        Yields:
            sqlite3.Connection: Configured database connection
        """
        conn = None
        attempt = 0
        
        while attempt < DatabaseUpdater.MAX_RETRY_ATTEMPTS:
            try:
                conn = sqlite3.connect(
                    db_file_path, 
                    timeout=timeout,
                    isolation_level=None  # Autocommit mode disabled for manual transaction control
                )
                
                # Configure connection for better concurrency and performance
                conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
                conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                
                yield conn
                break
                
            except sqlite3.OperationalError as e:
                attempt += 1
                if "database is locked" in str(e).lower() and attempt < DatabaseUpdater.MAX_RETRY_ATTEMPTS:
                    logger.warning(f"Database locked, retrying in {DatabaseUpdater.RETRY_DELAY}s (attempt {attempt})")
                    time.sleep(DatabaseUpdater.RETRY_DELAY)
                    continue
                else:
                    raise
                    
            except Exception as e:
                raise
                
            finally:
                if conn:
                    conn.close()
    
    @staticmethod
    def update_single_result(db_file_path: str, record_id: int, new_result: str,
                           timeout: float = DEFAULT_TIMEOUT) -> bool:
        """
        Update a single record with a new result.
        
        This is a convenience method for updating individual records.
        For bulk updates, use update_results() instead.
        
        Args:
            db_file_path: Path to the SQLite database file
            record_id: ID of the record to update
            new_result: New result value
            timeout: Database connection timeout in seconds
            
        Returns:
            bool: True if update was successful, False otherwise
            
        Raises:
            sqlite3.Error: If database operations fail
            ValueError: If input parameters are invalid
        """
        if not isinstance(record_id, int) or record_id <= 0:
            raise ValueError("record_id must be a positive integer")
        
        if not isinstance(new_result, str):
            raise ValueError("new_result must be a string")
        
        results_map = {record_id: new_result}
        failed_updates = DatabaseUpdater.update_results(db_file_path, results_map, timeout)
        
        return record_id not in failed_updates
    
    @staticmethod
    def verify_updates(db_file_path: str, expected_results: Dict[int, str],
                      timeout: float = DEFAULT_TIMEOUT) -> Tuple[List[int], List[int]]:
        """
        Verify that database updates were applied correctly.
        
        Args:
            db_file_path: Path to the SQLite database file
            expected_results: Dictionary mapping record IDs to expected result values
            timeout: Database connection timeout in seconds
            
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List of record IDs that match expected results
                - List of record IDs that don't match expected results
                
        Raises:
            sqlite3.Error: If database operations fail
        """
        if not expected_results:
            return [], []
        
        verified_records = []
        mismatched_records = []
        
        try:
            with DatabaseUpdater._get_database_connection(db_file_path, timeout) as conn:
                cursor = conn.cursor()
                
                # Check each expected result
                for record_id, expected_result in expected_results.items():
                    cursor.execute("SELECT result FROM llm_results WHERE id = ?", (record_id,))
                    row = cursor.fetchone()
                    
                    if row is None:
                        logger.warning(f"Record {record_id} not found in database")
                        mismatched_records.append(record_id)
                    elif row[0] == expected_result:
                        verified_records.append(record_id)
                    else:
                        logger.warning(f"Record {record_id}: expected '{expected_result}', got '{row[0]}'")
                        mismatched_records.append(record_id)
                
                logger.info(f"Verification complete: {len(verified_records)} correct, {len(mismatched_records)} mismatched")
                
        except sqlite3.Error as e:
            logger.error(f"Database error during verification: {str(e)}")
            raise
        
        return verified_records, mismatched_records
    
    @staticmethod
    def get_update_statistics(db_file_path: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """
        Get statistics about the database update status.
        
        Args:
            db_file_path: Path to the SQLite database file
            timeout: Database connection timeout in seconds
            
        Returns:
            dict: Statistics including total records, complete records, etc.
            
        Raises:
            sqlite3.Error: If database operations fail
        """
        try:
            with DatabaseUpdater._get_database_connection(db_file_path, timeout) as conn:
                cursor = conn.cursor()
                
                # Get total record count
                cursor.execute("SELECT COUNT(*) FROM llm_results")
                total_records = cursor.fetchone()[0]
                
                # Get complete record count (non-NULL, non-empty, non-"NA")
                cursor.execute("""
                    SELECT COUNT(*) FROM llm_results 
                    WHERE result IS NOT NULL AND result != '' AND result != 'NA'
                """)
                complete_records = cursor.fetchone()[0]
                
                # Get incomplete record counts by type
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result IS NULL")
                null_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = ''")
                empty_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = 'NA'")
                na_records = cursor.fetchone()[0]
                
                incomplete_records = null_records + empty_records + na_records
                
                return {
                    'total_records': total_records,
                    'complete_records': complete_records,
                    'incomplete_records': incomplete_records,
                    'null_records': null_records,
                    'empty_records': empty_records,
                    'na_records': na_records,
                    'completion_rate': (complete_records / total_records * 100) if total_records > 0 else 0
                }
                
        except sqlite3.Error as e:
            logger.error(f"Database error getting statistics: {str(e)}")
            raise