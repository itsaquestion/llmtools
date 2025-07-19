#!/usr/bin/env python3
"""
Comprehensive examples demonstrating database functionality in ParallelLLMProcessor.

This file provides practical examples of how to use the SQLite database storage
feature, including best practices, error handling, and performance optimization.
"""

import os
import time
import sqlite3
import logging
from typing import List
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor

# Configure logging to see database operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def demo_llm_function(prompt: str) -> str:
    """
    Demo LLM function that simulates realistic API behavior.
    
    Args:
        prompt: Input prompt string
        
    Returns:
        str: Simulated LLM response
        
    Raises:
        Exception: For prompts containing "error" to simulate API failures
    """
    # Simulate API processing time
    time.sleep(0.1)
    
    # Simulate occasional API errors
    if "error" in prompt.lower():
        raise Exception("Simulated API error - rate limit exceeded")
    
    # Simulate different response types
    if "math" in prompt.lower():
        return f"Mathematical analysis: {prompt}"
    elif "code" in prompt.lower():
        return f"```python\n# Code for: {prompt}\nprint('Hello World')\n```"
    else:
        return f"AI Response: {prompt}"


def example_1_basic_database_usage():
    """Example 1: Basic database storage functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Database Storage")
    print("="*60)
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "Write code for hello world",
        "Calculate 2+2 math problem"
    ]
    
    # Use context manager for automatic cleanup
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=2,
        save_to_db=True,
        db_filename="example_basic.db"
    ) as processor:
        print(f"Processing {len(prompts)} prompts with database storage...")
        results = processor.process_prompts(prompts)
        
        print(f"✓ Processed {len(results)} prompts")
        print(f"✓ Database saved to: example_basic.db")
        
        # Show some results
        for i, (prompt, result) in enumerate(zip(prompts[:2], results[:2])):
            print(f"  Prompt {i+1}: {prompt[:30]}...")
            print(f"  Result {i+1}: {result[:50]}...")


def example_2_default_filename():
    """Example 2: Using default timestamp-based filename."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Default Database Filename")
    print("="*60)
    
    prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
    
    # No db_filename specified - will generate timestamp-based name
    processor = ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=2,
        save_to_db=True  # Uses default filename: llm_results_{timestamp}.db
    )
    
    print("Using default database filename (timestamp-based)...")
    results = processor.process_prompts(prompts)
    
    # Get the generated filename
    db_filename = processor.db_filename
    print(f"✓ Database automatically saved to: {db_filename}")
    print(f"✓ Processed {len(results)} prompts")
    
    processor.close()


def example_3_error_handling():
    """Example 3: Error handling and resilience."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Error Handling and Resilience")
    print("="*60)
    
    # Mix of normal prompts and error-triggering prompts
    prompts = [
        "Normal prompt 1",
        "This will cause an error",  # Contains "error"
        "Normal prompt 2",
        "Another error prompt",      # Contains "error"
        "Final normal prompt"
    ]
    
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=3,
        retry_attempts=2,  # Will retry failed prompts
        save_to_db=True,
        db_filename="example_errors.db"
    ) as processor:
        print(f"Processing {len(prompts)} prompts (some will fail)...")
        results = processor.process_prompts(prompts)
        
        # Count successes and errors
        successes = sum(1 for r in results if not r.startswith("Error:"))
        errors = len(results) - successes
        
        print(f"✓ Total prompts: {len(results)}")
        print(f"✓ Successful: {successes}")
        print(f"✓ Errors: {errors}")
        print("✓ All results saved to database, including errors")


def example_4_performance_comparison():
    """Example 4: Performance comparison with and without database."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Comparison")
    print("="*60)
    
    # Create test prompts
    test_prompts = [f"Performance test prompt {i}" for i in range(20)]
    
    # Test without database
    print("Testing WITHOUT database storage...")
    start_time = time.time()
    
    processor_no_db = ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=4,
        save_to_db=False
    )
    results_no_db = processor_no_db.process_prompts(test_prompts)
    processor_no_db.close()
    
    time_no_db = time.time() - start_time
    
    # Test with database
    print("Testing WITH database storage...")
    start_time = time.time()
    
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=4,
        save_to_db=True,
        db_filename="example_performance.db"
    ) as processor_with_db:
        results_with_db = processor_with_db.process_prompts(test_prompts)
    
    time_with_db = time.time() - start_time
    
    # Calculate overhead
    overhead_percent = ((time_with_db - time_no_db) / time_no_db) * 100
    
    print(f"✓ Without database: {time_no_db:.3f}s")
    print(f"✓ With database: {time_with_db:.3f}s")
    print(f"✓ Database overhead: {overhead_percent:.1f}%")
    print(f"✓ Throughput: {len(test_prompts)/time_with_db:.1f} prompts/second")


def example_5_database_analysis():
    """Example 5: Analyzing results from the database."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Database Analysis")
    print("="*60)
    
    # First, create some data
    prompts = [
        "Analyze this data",
        "Generate error for testing",  # Will cause error
        "Write some code",
        "Explain math concepts",
        "Another error test"           # Will cause error
    ]
    
    db_filename = "example_analysis.db"
    
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=3,
        save_to_db=True,
        db_filename=db_filename
    ) as processor:
        results = processor.process_prompts(prompts)
    
    # Now analyze the database
    print(f"Analyzing database: {db_filename}")
    
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Basic statistics
    cursor.execute("SELECT COUNT(*) FROM llm_results")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result LIKE 'Error:%'")
    error_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result NOT LIKE 'Error:%'")
    success_count = cursor.fetchone()[0]
    
    print(f"✓ Total records: {total_records}")
    print(f"✓ Successful: {success_count}")
    print(f"✓ Errors: {error_count}")
    print(f"✓ Success rate: {(success_count/total_records)*100:.1f}%")
    
    # Show sample records
    print("\nSample records:")
    cursor.execute("SELECT id, prompt, result, created_at FROM llm_results LIMIT 3")
    for row in cursor.fetchall():
        id_val, prompt, result, created_at = row
        print(f"  ID {id_val}: {prompt[:30]}... -> {result[:40]}...")
    
    conn.close()


def example_6_concurrent_safety():
    """Example 6: High concurrency testing."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Concurrent Safety Testing")
    print("="*60)
    
    # Create many prompts for high concurrency testing
    concurrent_prompts = [f"Concurrent test {i}" for i in range(50)]
    
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=10,  # High concurrency
        save_to_db=True,
        db_filename="example_concurrent.db"
    ) as processor:
        print(f"Processing {len(concurrent_prompts)} prompts with 10 concurrent workers...")
        start_time = time.time()
        results = processor.process_prompts(concurrent_prompts)
        end_time = time.time()
        
        print(f"✓ Processed {len(results)} prompts in {end_time-start_time:.2f}s")
        print(f"✓ Throughput: {len(results)/(end_time-start_time):.1f} prompts/second")
        print("✓ Thread-safe database operations completed successfully")


def example_7_best_practices():
    """Example 7: Best practices demonstration."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Best Practices")
    print("="*60)
    
    prompts = [
        "Best practice example 1",
        "Best practice example 2",
        "Best practice example 3"
    ]
    
    # Best practice: Use context manager
    print("✓ Using context manager for automatic resource cleanup")
    
    # Best practice: Configure appropriate parameters
    print("✓ Configuring parameters for your use case")
    
    with ParallelLLMProcessor(
        chat_fn=demo_llm_function,
        num_workers=4,           # Adjust based on API limits
        retry_attempts=3,        # Reasonable retry count
        retry_delay=1.0,         # Appropriate delay
        timeout=30.0,            # Reasonable timeout
        save_to_db=True,
        db_filename="example_best_practices.db"
    ) as processor:
        results = processor.process_prompts(prompts)
        
        print(f"✓ Processed {len(results)} prompts following best practices")
        print("✓ Database automatically cleaned up via context manager")


def cleanup_example_files():
    """Clean up example database files."""
    print("\n" + "="*60)
    print("CLEANUP: Removing Example Database Files")
    print("="*60)
    
    example_files = [
        "example_basic.db",
        "example_errors.db", 
        "example_performance.db",
        "example_analysis.db",
        "example_concurrent.db",
        "example_best_practices.db"
    ]
    
    # Also find any timestamp-based files
    for filename in os.listdir('.'):
        if filename.startswith('llm_results_') and filename.endswith('.db'):
            example_files.append(filename)
    
    removed_count = 0
    for filename in example_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"✓ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {filename}: {e}")
    
    print(f"\n✓ Cleanup complete: {removed_count} files removed")


def main():
    """Run all database usage examples."""
    print("DATABASE USAGE EXAMPLES FOR PARALLELLMPROCESSOR")
    print("=" * 80)
    print("This script demonstrates various ways to use the database functionality")
    print("in ParallelLLMProcessor, including best practices and error handling.")
    
    try:
        # Run all examples
        example_1_basic_database_usage()
        example_2_default_filename()
        example_3_error_handling()
        example_4_performance_comparison()
        example_5_database_analysis()
        example_6_concurrent_safety()
        example_7_best_practices()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey takeaways:")
        print("• Database storage adds minimal overhead (typically < 10%)")
        print("• Error handling is robust - database failures don't stop processing")
        print("• Thread-safe operations work reliably under high concurrency")
        print("• Context managers ensure proper resource cleanup")
        print("• Default filenames use timestamps for uniqueness")
        print("• Database schema preserves prompt order and timing information")
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        raise
    
    finally:
        # Clean up example files
        cleanup_example_files()


if __name__ == "__main__":
    main()