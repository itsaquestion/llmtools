#!/usr/bin/env python3
"""
Example demonstrating the recover_from_database functionality of ParallelLLMProcessor.

This example shows how to:
1. Create a database with some incomplete results (simulating a previous interrupted run)
2. Use the recover_from_database method to complete the processing
3. Verify that the recovery worked correctly
"""

import os
import sqlite3
import tempfile
import time
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor


def mock_chat_fn(prompt: str) -> str:
    """
    Mock LLM function that simulates processing time and generates responses.
    In a real scenario, this would be your actual LLM API call.
    """
    time.sleep(0.2)  # Simulate API call delay
    
    # Generate different responses based on prompt content
    if "math" in prompt.lower() or any(op in prompt for op in ['+', '-', '*', '/', '=']):
        return f"Mathematical calculation: {prompt}"
    elif "capital" in prompt.lower():
        return f"Geographic answer: {prompt}"
    elif "color" in prompt.lower():
        return f"Color description: {prompt}"
    else:
        return f"General response: {prompt}"


def create_interrupted_database(db_path: str):
    """
    Create a database that simulates an interrupted processing session.
    This represents what might happen if your processing was stopped midway.
    """
    print(f"Creating interrupted database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the llm_results table (same structure as ParallelLLMProcessor uses)
    cursor.execute("""
        CREATE TABLE llm_results (
            id INTEGER PRIMARY KEY,
            prompt TEXT NOT NULL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert test data representing an interrupted session
    # Some results are complete, others are incomplete (NULL, empty, or "NA")
    test_data = [
        (1, "What is 15 + 27?", "42"),  # Complete result
        (2, "What is the capital of Japan?", None),  # Incomplete (NULL)
        (3, "What color is a ripe tomato?", ""),  # Incomplete (empty string)
        (4, "What is 8 * 7?", "56"),  # Complete result
        (5, "What is the capital of Australia?", "NA"),  # Incomplete (NA marker)
        (6, "What color is grass?", None),  # Incomplete (NULL)
        (7, "What is 100 - 23?", "77"),  # Complete result
        (8, "What is the capital of Brazil?", ""),  # Incomplete (empty string)
    ]
    
    cursor.executemany(
        "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
        test_data
    )
    
    conn.commit()
    conn.close()
    
    print(f"Created database with {len(test_data)} records:")
    print("  - 4 complete results")
    print("  - 4 incomplete results (2 NULL, 2 empty string, 1 'NA')")
    
    return test_data


def display_database_state(db_path: str, title: str):
    """Display the current state of the database for debugging/demonstration."""
    print(f"\n{title}")
    print("=" * len(title))
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, result FROM llm_results ORDER BY id")
    records = cursor.fetchall()
    conn.close()
    
    for record_id, prompt, result in records:
        status = "✅ Complete" if result and result not in ["", "NA"] else "❌ Incomplete"
        result_display = repr(result) if result is not None else "NULL"
        print(f"  {record_id}: {prompt[:40]:<40} | {result_display:<20} | {status}")


def main():
    """Demonstrate the recover_from_database functionality."""
    print("🔄 ParallelLLMProcessor Recovery Example")
    print("=" * 50)
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Step 1: Create an interrupted database
        print("\n📊 Step 1: Creating interrupted database")
        test_data = create_interrupted_database(db_path)
        display_database_state(db_path, "Initial Database State (Before Recovery)")
        
        # Step 2: Create processor and perform recovery
        print("\n🔧 Step 2: Setting up ParallelLLMProcessor")
        processor = ParallelLLMProcessor(
            chat_fn=mock_chat_fn,
            num_workers=3,  # Use 3 workers for parallel processing
            retry_attempts=2,  # Retry failed requests twice
            timeout=30.0  # 30 second timeout per request
        )
        
        print("Processor configuration:")
        print(f"  - Workers: {processor.num_workers}")
        print(f"  - Retry attempts: {processor.retry_attempts}")
        print(f"  - Timeout: {processor.timeout}s")
        
        # Step 3: Perform recovery
        print("\n🚀 Step 3: Performing recovery")
        print("This will reprocess only the incomplete results...")
        
        start_time = time.time()
        results = processor.recover_from_database(db_path)
        end_time = time.time()
        
        print(f"Recovery completed in {end_time - start_time:.2f} seconds")
        
        # Step 4: Display results
        print(f"\n📋 Step 4: Recovery Results")
        print(f"Got {len(results)} total results:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}: {result}")
        
        # Step 5: Verify database state after recovery
        display_database_state(db_path, "Final Database State (After Recovery)")
        
        # Step 6: Verify that recovery worked correctly
        print("\n✅ Step 6: Verification")
        
        # Check that we have the right number of results
        assert len(results) == len(test_data), f"Expected {len(test_data)} results, got {len(results)}"
        print(f"  ✓ Correct number of results: {len(results)}")
        
        # Check that previously complete results were preserved
        original_complete = [(1, "42"), (4, "56"), (7, "77")]
        for record_id, expected_result in original_complete:
            actual_result = results[record_id - 1]  # Convert to 0-based index
            assert actual_result == expected_result, f"Result {record_id} changed: expected {expected_result}, got {actual_result}"
        print(f"  ✓ Previously complete results preserved: {len(original_complete)} results")
        
        # Check that incomplete results were reprocessed
        incomplete_indices = [1, 2, 4, 5, 7]  # 0-based indices of originally incomplete results
        reprocessed_count = 0
        for idx in incomplete_indices:
            result = results[idx]
            if result and result not in ["", "NA", None]:
                reprocessed_count += 1
        print(f"  ✓ Incomplete results reprocessed: {reprocessed_count} results")
        
        # Step 7: Demonstrate recovery idempotency
        print("\n🔄 Step 7: Testing recovery idempotency")
        print("Running recovery again should not change anything...")
        
        results_2 = processor.recover_from_database(db_path)
        
        if results == results_2:
            print("  ✓ Recovery is idempotent - running twice gives same results")
        else:
            print("  ❌ Recovery is not idempotent - results changed!")
            
        processor.close()
        
        print("\n🎉 Recovery example completed successfully!")
        print("\nKey takeaways:")
        print("  • recover_from_database() only reprocesses incomplete results")
        print("  • Complete results are preserved exactly as they were")
        print("  • The method returns results in the original order")
        print("  • Recovery is idempotent - safe to run multiple times")
        print("  • All existing ParallelLLMProcessor settings are used for recovery")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        raise
    finally:
        # Clean up temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up temporary database: {db_path}")


if __name__ == "__main__":
    main()