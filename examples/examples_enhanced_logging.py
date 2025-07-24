#!/usr/bin/env python3
"""
Enhanced Logging and Monitoring Example

This example demonstrates the comprehensive logging and monitoring functionality
implemented for task 6 of the parallel processor recovery feature.

Features demonstrated:
1. Detailed recovery process logging with step-by-step progress
2. Enhanced progress bars with dynamic status updates
3. Comprehensive recovery operation summary
4. Clear error messages with context information
5. Database analysis with detailed statistics
6. Processing configuration logging
7. Performance timing information
"""

import logging
import sqlite3
import os
import sys
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from llmtools.parallel_llm_processor import ParallelLLMProcessor

def setup_logging():
    """Setup comprehensive logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/recovery_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to DEBUG for more detailed output
    logging.getLogger('src.llmtools.recovery_processor').setLevel(logging.DEBUG)
    logging.getLogger('src.llmtools.recovery_analyzer').setLevel(logging.DEBUG)
    logging.getLogger('src.llmtools.database_updater').setLevel(logging.DEBUG)

def mock_chat_fn_with_delays(prompt: str) -> str:
    """Mock LLM function that simulates realistic processing times and occasional failures."""
    # Simulate variable processing times
    if "complex" in prompt.lower():
        time.sleep(0.3)  # Longer processing for complex prompts
    elif "simple" in prompt.lower():
        time.sleep(0.1)  # Quick processing for simple prompts
    else:
        time.sleep(0.2)  # Standard processing time
    
    # Simulate occasional failures for demonstration
    if "error" in prompt.lower():
        raise Exception("Simulated LLM API error")
    elif "timeout" in prompt.lower():
        time.sleep(10)  # This will cause a timeout
        return "This should timeout"
    elif "retry" in prompt.lower() and time.time() % 3 < 1:
        raise Exception("Intermittent failure - will succeed on retry")
    
    return f"Processed response for: {prompt}"

def create_comprehensive_test_database(db_path: str):
    """Create a comprehensive test database with various scenarios."""
    print("Creating comprehensive test database...")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database with comprehensive test data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE llm_results (
            id INTEGER PRIMARY KEY,
            prompt TEXT NOT NULL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert comprehensive test data
    test_data = [
        # Complete records (should not be reprocessed)
        (1, "Simple question: What is 2+2?", "4"),
        (2, "What is the capital of France?", "Paris"),
        (3, "Explain basic math", "Mathematics is the study of numbers..."),
        
        # NULL records (need recovery)
        (4, "Complex analysis question", None),
        (5, "Simple calculation task", None),
        (6, "This will cause an error", None),
        
        # Empty string records (need recovery)
        (7, "Complex reasoning problem", ""),
        (8, "Simple logic question", ""),
        (9, "This will timeout during processing", ""),
        
        # "NA" records (need recovery)
        (10, "Complex mathematical proof", "NA"),
        (11, "Simple arithmetic", "NA"),
        (12, "This will retry and succeed", "NA"),
        
        # More complete records
        (13, "What is AI?", "Artificial Intelligence is..."),
        (14, "Hello world", "Hello! How can I help you?"),
        (15, "Final question", "Final answer provided"),
        
        # Additional incomplete records for batch processing demo
        (16, "Batch processing test 1", None),
        (17, "Batch processing test 2", ""),
        (18, "Batch processing test 3", "NA"),
        (19, "Complex batch item", None),
        (20, "Simple batch item", ""),
    ]
    
    cursor.executemany(
        "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
        test_data
    )
    
    conn.commit()
    conn.close()
    
    # Calculate statistics
    total_records = len(test_data)
    complete_records = sum(1 for _, _, result in test_data if result and result != "NA")
    incomplete_records = total_records - complete_records
    
    print(f"✓ Comprehensive test database created:")
    print(f"  - Total records: {total_records}")
    print(f"  - Complete records: {complete_records}")
    print(f"  - Incomplete records: {incomplete_records}")
    print(f"  - Initial completion rate: {(complete_records / total_records * 100):.1f}%")
    print(f"  - NULL records: {sum(1 for _, _, result in test_data if result is None)}")
    print(f"  - Empty string records: {sum(1 for _, _, result in test_data if result == '')}")
    print(f"  - 'NA' records: {sum(1 for _, _, result in test_data if result == 'NA')}")

def demonstrate_enhanced_logging():
    """Main demonstration function."""
    print("=" * 80)
    print("ENHANCED LOGGING AND MONITORING DEMONSTRATION")
    print("=" * 80)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    # Setup logging
    setup_logging()
    
    # Test database path
    test_db = "enhanced_logging_demo.db"
    
    try:
        # Step 1: Create comprehensive test database
        create_comprehensive_test_database(test_db)
        print()
        
        # Step 2: Initialize processor with specific configuration for demonstration
        print("Initializing ParallelLLMProcessor with demonstration configuration...")
        processor = ParallelLLMProcessor(
            chat_fn=mock_chat_fn_with_delays,
            num_workers=3,  # Use 3 workers for good parallelism demonstration
            retry_attempts=3,  # Allow retries to show retry logging
            retry_delay=0.5,   # Short delay for demo
            timeout=2.0        # Short timeout to demonstrate timeout handling
        )
        print("✓ Processor initialized with enhanced logging configuration")
        print()
        
        # Step 3: Run recovery with comprehensive logging demonstration
        print("Starting recovery process with enhanced logging and monitoring...")
        print("This will demonstrate:")
        print("  - Step-by-step recovery process logging")
        print("  - Database analysis with detailed statistics")
        print("  - Progress bars with dynamic status updates")
        print("  - Error handling and retry logging")
        print("  - Comprehensive recovery summary")
        print("  - Performance timing information")
        print()
        
        # Run the recovery process
        results = processor.recover_from_database(test_db)
        
        print()
        print("=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Display final results analysis
        complete_results = sum(1 for r in results if r and r != "NA" and not r.startswith("Reprocessing failed:") and not r.startswith("Error:"))
        error_results = sum(1 for r in results if r and (r.startswith("Reprocessing failed:") or r.startswith("Error:")))
        incomplete_results = len(results) - complete_results - error_results
        
        print(f"Final Results Analysis:")
        print(f"  - Total results: {len(results)}")
        print(f"  - Complete results: {complete_results}")
        print(f"  - Error results: {error_results}")
        print(f"  - Still incomplete: {incomplete_results}")
        print(f"  - Final completion rate: {(complete_results / len(results) * 100):.1f}%")
        
        # Show sample results
        print(f"\nSample Results (first 10):")
        for i, result in enumerate(results[:10], 1):
            if result and result != "NA" and not result.startswith("Reprocessing failed:") and not result.startswith("Error:"):
                status = "✅ COMPLETE"
            elif result and (result.startswith("Reprocessing failed:") or result.startswith("Error:")):
                status = "❌ ERROR"
            else:
                status = "⚠️  INCOMPLETE"
            
            result_preview = result[:60] + "..." if result and len(result) > 60 else result
            print(f"  {i:2d}. {status}: {result_preview}")
        
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more results")
        
        processor.close()
        
        print(f"\n📊 Check the log file in the 'logs' directory for complete detailed logging output")
        
    except Exception as e:
        print(f"❌ Demonstration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"\n🧹 Cleaned up test database: {test_db}")
    
    print(f"\nDemonstration completed at: {datetime.now()}")
    return 0

if __name__ == "__main__":
    sys.exit(demonstrate_enhanced_logging())