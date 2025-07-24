"""
Recovery analyzer module for analyzing database records and identifying incomplete results.

This module provides the RecoveryAnalyzer class which is responsible for:
- Analyzing SQLite database records to identify incomplete or failed results
- Extracting failed records and existing results from the database
- Determining which results need to be reprocessed
"""

import sqlite3
import logging
from typing import List, Tuple, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class RecoveryAnalyzer:
    """
    Analyzer for identifying incomplete results in the database and extracting recovery data.
    
    This class provides static methods for analyzing database records to identify
    which prompts need to be reprocessed based on their result status.
    """
    
    @staticmethod
    def is_result_incomplete(result: Any) -> bool:
        """
        Determine if a result is incomplete and needs to be reprocessed.
        
        A result is considered incomplete if it is:
        - None (NULL in database)
        - Empty string ("")
        - The string "NA"
        
        Args:
            result: The result value to check (can be None, str, or other types)
            
        Returns:
            bool: True if the result is incomplete and needs reprocessing, False otherwise
        """
        # Handle None/NULL values
        if result is None:
            return True
        
        # Handle string values
        if isinstance(result, str):
            # Empty string or "NA" are considered incomplete
            return result == "" or result == "NA"
        
        # For any other type, consider it complete
        # This handles cases where result might be a number or other data type
        return False
    
    @staticmethod
    def analyze_database(db_file_path: str) -> Tuple[List[Tuple[int, str]], List[str]]:
        """
        Analyze the database to identify failed records and extract all existing results.
        
        This method:
        1. Connects to the SQLite database
        2. Reads all records from the llm_results table ordered by id
        3. Identifies records with incomplete results
        4. Returns both failed records (for reprocessing) and all existing results (for final output)
        
        Args:
            db_file_path: Path to the SQLite database file
            
        Returns:
            Tuple[List[Tuple[int, str]], List[str]]: A tuple containing:
                - List of failed records as (id, prompt) tuples
                - List of all existing results in order (with None for incomplete results)
                
        Raises:
            sqlite3.Error: If database operations fail
            FileNotFoundError: If database file doesn't exist (handled by sqlite3)
            ValueError: If database structure is invalid
        """
        # Enhanced logging: Analysis start
        logger.debug(f"Starting database analysis: {db_file_path}")
        
        failed_records = []
        existing_results = []
        
        # Statistics tracking
        null_count = 0
        empty_string_count = 0
        na_count = 0
        complete_count = 0
        
        try:
            import time
            analysis_start_time = time.time()
            
            # Connect to database with optimized settings
            with sqlite3.connect(db_file_path, timeout=30.0) as conn:
                # Performance optimizations for large databases
                conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/performance
                conn.execute("PRAGMA cache_size=10000")  # Larger cache for better performance
                conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp storage
                
                cursor = conn.cursor()
                
                # Enhanced logging: Database connection established
                logger.debug("Database connection established with performance optimizations")
                
                # Check if llm_results table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='llm_results'
                """)
                
                if not cursor.fetchone():
                    logger.error("Database validation failed: 'llm_results' table not found")
                    raise ValueError("Database does not contain 'llm_results' table")
                
                logger.debug("✓ Database table structure validated")
                
                # Get total record count first for progress tracking
                cursor.execute("SELECT COUNT(*) FROM llm_results")
                total_records = cursor.fetchone()[0]
                
                if total_records == 0:
                    logger.info("Database contains no records")
                    return [], []
                
                logger.debug(f"Found {total_records} total records to analyze")
                
                # Memory optimization: Use streaming for large datasets
                if total_records > 10000:
                    logger.info(f"Large dataset detected ({total_records} records), using streaming analysis")
                    return RecoveryAnalyzer._analyze_database_streaming(conn, cursor, total_records)
                
                # Fetch all records ordered by id to maintain original order
                cursor.execute("""
                    SELECT id, prompt, result 
                    FROM llm_results 
                    ORDER BY id
                """)
                
                records = cursor.fetchall()
                analysis_duration = time.time() - analysis_start_time
                
                logger.debug(f"Records retrieved in {analysis_duration:.3f} seconds")
                
                # Process each record with enhanced tracking
                for i, (record_id, prompt, result) in enumerate(records, 1):
                    # Check if result is incomplete
                    if RecoveryAnalyzer.is_result_incomplete(result):
                        failed_records.append((record_id, prompt))
                        existing_results.append(None)  # Placeholder for incomplete result
                        
                        # Track specific types of incomplete results
                        if result is None:
                            null_count += 1
                            logger.debug(f"Record {record_id}: NULL result")
                        elif result == "":
                            empty_string_count += 1
                            logger.debug(f"Record {record_id}: Empty string result")
                        elif result == "NA":
                            na_count += 1
                            logger.debug(f"Record {record_id}: 'NA' result")
                    else:
                        existing_results.append(result)
                        complete_count += 1
                        logger.debug(f"Record {record_id}: Complete result ({len(str(result))} chars)")
                    
                    # Log progress for large datasets
                    if total_records > 1000 and i % (total_records // 10) == 0:
                        progress_pct = (i / total_records * 100)
                        logger.debug(f"Analysis progress: {i}/{total_records} records processed ({progress_pct:.0f}%)")
                
                total_analysis_time = time.time() - analysis_start_time
                
                # Enhanced logging: Analysis results with detailed breakdown
                logger.info(f"Database analysis completed in {total_analysis_time:.3f} seconds")
                logger.info(f"Analysis results:")
                logger.info(f"  - Total records: {len(records)}")
                logger.info(f"  - Complete records: {complete_count}")
                logger.info(f"  - Incomplete records: {len(failed_records)}")
                logger.info(f"    • NULL results: {null_count}")
                logger.info(f"    • Empty string results: {empty_string_count}")
                logger.info(f"    • 'NA' results: {na_count}")
                logger.info(f"  - Completion rate: {(complete_count / len(records) * 100):.1f}%")
                
                if len(failed_records) > 0:
                    logger.info(f"Found {len(failed_records)} records requiring reprocessing")
                    
                    # Log sample of failed records for debugging
                    if len(failed_records) <= 5:
                        logger.debug("Failed records:")
                        for record_id, prompt in failed_records:
                            logger.debug(f"  - Record {record_id}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
                    else:
                        logger.debug("Sample of failed records (first 3):")
                        for record_id, prompt in failed_records[:3]:
                            logger.debug(f"  - Record {record_id}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
                        logger.debug(f"  ... and {len(failed_records) - 3} more records")
                else:
                    logger.info("✅ All records are complete - no reprocessing needed")
                
                return failed_records, existing_results
                
        except sqlite3.OperationalError as e:
            if "no such file or directory" in str(e).lower() or "unable to open database file" in str(e).lower():
                logger.error(f"Database file not found: {db_file_path}")
                raise FileNotFoundError(f"Database file not found: {db_file_path}")
            else:
                logger.error(f"Database operational error: {str(e)}")
                raise sqlite3.Error(f"Database operation failed: {str(e)}")
                
        except ValueError as e:
            # Re-raise ValueError directly (for table structure issues)
            logger.error(f"Database structure error: {str(e)}")
            raise
            
        except sqlite3.Error as e:
            logger.error(f"Database error during analysis: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during database analysis: {str(e)}")
            raise RuntimeError(f"Database analysis failed: {str(e)}")
    
    @staticmethod
    def _analyze_database_streaming(conn: sqlite3.Connection, cursor: sqlite3.Cursor, total_records: int) -> Tuple[List[Tuple[int, str]], List[str]]:
        """
        Analyze database using streaming approach for large datasets to optimize memory usage.
        
        Args:
            conn: Database connection
            cursor: Database cursor
            total_records: Total number of records
            
        Returns:
            Tuple[List[Tuple[int, str]], List[str]]: Failed records and existing results
        """
        failed_records = []
        existing_results = []
        
        # Statistics tracking
        null_count = 0
        empty_string_count = 0
        na_count = 0
        complete_count = 0
        
        # Process records in chunks to manage memory
        chunk_size = 1000
        processed_count = 0
        
        logger.info(f"Processing {total_records} records in chunks of {chunk_size}")
        
        # Use LIMIT and OFFSET for chunked processing
        for offset in range(0, total_records, chunk_size):
            cursor.execute("""
                SELECT id, prompt, result 
                FROM llm_results 
                ORDER BY id
                LIMIT ? OFFSET ?
            """, (chunk_size, offset))
            
            chunk_records = cursor.fetchall()
            
            for record_id, prompt, result in chunk_records:
                # Check if result is incomplete
                if RecoveryAnalyzer.is_result_incomplete(result):
                    failed_records.append((record_id, prompt))
                    existing_results.append(None)
                    
                    # Track specific types of incomplete results
                    if result is None:
                        null_count += 1
                    elif result == "":
                        empty_string_count += 1
                    elif result == "NA":
                        na_count += 1
                else:
                    existing_results.append(result)
                    complete_count += 1
                
                processed_count += 1
            
            # Log progress every 10 chunks or at the end
            if (offset // chunk_size + 1) % 10 == 0 or processed_count >= total_records:
                progress_pct = (processed_count / total_records * 100)
                logger.info(f"Streaming analysis progress: {processed_count}/{total_records} records ({progress_pct:.1f}%)")
        
        logger.info(f"Streaming analysis completed:")
        logger.info(f"  - Complete records: {complete_count}")
        logger.info(f"  - Incomplete records: {len(failed_records)}")
        logger.info(f"    • NULL results: {null_count}")
        logger.info(f"    • Empty string results: {empty_string_count}")
        logger.info(f"    • 'NA' results: {na_count}")
        
        return failed_records, existing_results
    
    @staticmethod
    def get_database_summary(db_file_path: str) -> dict:
        """
        Get a summary of the database contents for logging and debugging.
        
        Args:
            db_file_path: Path to the SQLite database file
            
        Returns:
            dict: Summary containing total records, complete records, incomplete records, etc.
            
        Raises:
            sqlite3.Error: If database operations fail
        """
        try:
            failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_file_path)
            
            total_records = len(existing_results)
            incomplete_records = len(failed_records)
            complete_records = total_records - incomplete_records
            
            # Count different types of incomplete results
            null_count = 0
            empty_string_count = 0
            na_count = 0
            
            with sqlite3.connect(db_file_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Count NULL results
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result IS NULL")
                null_count = cursor.fetchone()[0]
                
                # Count empty string results
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = ''")
                empty_string_count = cursor.fetchone()[0]
                
                # Count "NA" results
                cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = 'NA'")
                na_count = cursor.fetchone()[0]
            
            return {
                'total_records': total_records,
                'complete_records': complete_records,
                'incomplete_records': incomplete_records,
                'null_results': null_count,
                'empty_string_results': empty_string_count,
                'na_results': na_count,
                'completion_rate': (complete_records / total_records * 100) if total_records > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to generate database summary: {str(e)}")
            raise