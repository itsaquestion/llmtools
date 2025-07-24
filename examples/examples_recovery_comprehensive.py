#!/usr/bin/env python3
"""
Comprehensive Recovery Examples for ParallelLLMProcessor

This example demonstrates various recovery scenarios and best practices:
1. Basic recovery from interrupted session
2. Recovery with different configurations
3. Batch recovery from multiple databases
4. Error handling and validation
5. Performance optimization strategies
6. Production-ready recovery patterns
"""

import os
import sqlite3
import tempfile
import time
import logging
import glob
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor


# Configure logging for detailed recovery monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mock_chat_fn(prompt: str) -> str:
    """
    Mock LLM function that simulates various response patterns.
    In production, replace this with your actual LLM API call.
    """
    time.sleep(0.1)  # Simulate API call delay
    
    # Simulate different response types based on prompt content
    if "math" in prompt.lower() or any(op in prompt for op in ['+', '-', '*', '/', '=']):
        return f"Mathematical result: {prompt}"
    elif "capital" in prompt.lower():
        return f"Geographic answer: {prompt}"
    elif "color" in prompt.lower():
        return f"Color description: {prompt}"
    elif "error" in prompt.lower():
        # Simulate occasional failures for testing
        raise Exception("Simulated API error")
    else:
        return f"General response: {prompt}"


def improved_chat_fn(prompt: str) -> str:
    """
    Improved LLM function with better error handling.
    Demonstrates how recovery can use an updated function.
    """
    try:
        return mock_chat_fn(prompt)
    except Exception as e:
        # Instead of failing, return a descriptive error message
        return f"Processing failed for '{prompt[:30]}...': {str(e)}"


def create_test_database(db_path: str, scenario: str = "mixed") -> List[Tuple[int, str, Optional[str]]]:
    """
    Create test databases with different failure scenarios.
    
    Args:
        db_path: Path for the database file
        scenario: Type of scenario to create
            - "mixed": Mix of complete and incomplete results
            - "mostly_failed": Mostly incomplete results
            - "few_failed": Few incomplete results
            - "all_complete": All results complete (no recovery needed)
    
    Returns:
        List of (id, prompt, result) tuples representing the test data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the llm_results table
    cursor.execute("""
        CREATE TABLE llm_results (
            id INTEGER PRIMARY KEY,
            prompt TEXT NOT NULL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Define test scenarios
    scenarios = {
        "mixed": [
            (1, "What is 15 + 27?", "42"),
            (2, "What is the capital of Japan?", None),
            (3, "What color is a ripe tomato?", ""),
            (4, "What is 8 * 7?", "56"),
            (5, "What is the capital of Australia?", "NA"),
            (6, "What color is grass?", None),
            (7, "What is 100 - 23?", "77"),
            (8, "What is the capital of Brazil?", ""),
            (9, "What color is the sky?", None),
            (10, "What is 12 / 4?", "3"),
        ],
        "mostly_failed": [
            (1, "Math problem 1", None),
            (2, "Math problem 2", ""),
            (3, "Math problem 3", "NA"),
            (4, "Math problem 4", None),
            (5, "Math problem 5", ""),
            (6, "Geography question 1", None),
            (7, "Geography question 2", "Complete answer"),
            (8, "Color question 1", None),
        ],
        "few_failed": [
            (1, "Question 1", "Complete answer 1"),
            (2, "Question 2", "Complete answer 2"),
            (3, "Question 3", "Complete answer 3"),
            (4, "Question 4", None),  # Only one failure
            (5, "Question 5", "Complete answer 5"),
            (6, "Question 6", "Complete answer 6"),
        ],
        "all_complete": [
            (1, "Question 1", "Complete answer 1"),
            (2, "Question 2", "Complete answer 2"),
            (3, "Question 3", "Complete answer 3"),
            (4, "Question 4", "Complete answer 4"),
            (5, "Question 5", "Complete answer 5"),
        ]
    }
    
    test_data = scenarios.get(scenario, scenarios["mixed"])
    
    cursor.executemany(
        "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
        test_data
    )
    
    conn.commit()
    conn.close()
    
    return test_data


def analyze_database_state(db_path: str) -> Dict:
    """Analyze the current state of a database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("SELECT COUNT(*) FROM llm_results")
        total_records = cursor.fetchone()[0]
        
        # Count incomplete results
        cursor.execute("""
            SELECT COUNT(*) FROM llm_results 
            WHERE result IS NULL OR result = '' OR result = 'NA'
        """)
        incomplete_records = cursor.fetchone()[0]
        
        # Count error results
        cursor.execute("""
            SELECT COUNT(*) FROM llm_results 
            WHERE result LIKE 'Error:%' OR result LIKE 'Processing failed%'
        """)
        error_records = cursor.fetchone()[0]
        
        complete_records = total_records - incomplete_records
        
        conn.close()
        
        return {
            'total': total_records,
            'complete': complete_records,
            'incomplete': incomplete_records,
            'errors': error_records,
            'completion_rate': (complete_records / total_records * 100) if total_records > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze database {db_path}: {e}")
        return {}


def display_database_state(db_path: str, title: str):
    """Display database state in a formatted way."""
    print(f"\n{title}")
    print("=" * len(title))
    
    stats = analyze_database_state(db_path)
    if not stats:
        print("  ❌ Could not analyze database")
        return
    
    print(f"  📊 Total records: {stats['total']}")
    print(f"  ✅ Complete records: {stats['complete']}")
    print(f"  ❌ Incomplete records: {stats['incomplete']}")
    print(f"  ⚠️  Error records: {stats['errors']}")
    print(f"  📈 Completion rate: {stats['completion_rate']:.1f}%")
    
    if stats['incomplete'] > 0:
        print(f"  🔄 Recovery recommended")
    else:
        print(f"  ✨ No recovery needed")


def example_1_basic_recovery():
    """Example 1: Basic recovery from interrupted session."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Recovery")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create interrupted database
        print("📊 Creating interrupted database...")
        create_test_database(db_path, "mixed")
        display_database_state(db_path, "Before Recovery")
        
        # Perform recovery
        print("\n🔄 Performing recovery...")
        with ParallelLLMProcessor(
            chat_fn=mock_chat_fn,
            num_workers=3,
            retry_attempts=2
        ) as processor:
            results = processor.recover_from_database(db_path)
        
        print(f"\n✅ Recovery completed! Got {len(results)} results")
        
        # Display final state
        display_database_state(db_path, "After Recovery")
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}: {result}")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def example_2_recovery_with_different_config():
    """Example 2: Recovery with different processor configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Recovery with Different Configuration")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create database with many failures
        print("📊 Creating database with many failures...")
        create_test_database(db_path, "mostly_failed")
        display_database_state(db_path, "Before Recovery")
        
        # First recovery attempt with basic configuration
        print("\n🔄 First recovery attempt (basic config)...")
        with ParallelLLMProcessor(
            chat_fn=mock_chat_fn,  # Original function that may fail
            num_workers=2,
            retry_attempts=1
        ) as processor:
            results_1 = processor.recover_from_database(db_path)
        
        display_database_state(db_path, "After First Recovery")
        
        # Second recovery attempt with improved configuration
        print("\n🔄 Second recovery attempt (improved config)...")
        with ParallelLLMProcessor(
            chat_fn=improved_chat_fn,  # Improved function with better error handling
            num_workers=4,
            retry_attempts=3,
            timeout=30.0
        ) as processor:
            results_2 = processor.recover_from_database(db_path)
        
        display_database_state(db_path, "After Second Recovery")
        
        print(f"\n📈 Recovery Progress:")
        print(f"  First attempt: {len(results_1)} results")
        print(f"  Second attempt: {len(results_2)} results")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def example_3_batch_recovery():
    """Example 3: Batch recovery from multiple databases."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Recovery")
    print("="*60)
    
    # Create multiple test databases
    temp_dir = tempfile.mkdtemp()
    db_files = []
    
    try:
        scenarios = ["mixed", "mostly_failed", "few_failed", "all_complete"]
        
        print("📊 Creating multiple test databases...")
        for i, scenario in enumerate(scenarios, 1):
            db_path = os.path.join(temp_dir, f"session_{i}_{scenario}.db")
            create_test_database(db_path, scenario)
            db_files.append(db_path)
            print(f"  Created: {os.path.basename(db_path)}")
        
        # Analyze all databases
        print(f"\n📋 Database Analysis:")
        for db_path in db_files:
            stats = analyze_database_state(db_path)
            name = os.path.basename(db_path)
            print(f"  {name}: {stats['incomplete']} incomplete of {stats['total']} total")
        
        # Perform batch recovery
        print(f"\n🔄 Performing batch recovery...")
        
        processor = ParallelLLMProcessor(
            chat_fn=improved_chat_fn,
            num_workers=3
        )
        
        recovery_results = {}
        
        for db_path in db_files:
            name = os.path.basename(db_path)
            stats_before = analyze_database_state(db_path)
            
            if stats_before['incomplete'] > 0:
                print(f"  🔄 Recovering {name}...")
                try:
                    results = processor.recover_from_database(db_path)
                    stats_after = analyze_database_state(db_path)
                    recovery_results[name] = {
                        'status': 'success',
                        'results_count': len(results),
                        'before': stats_before,
                        'after': stats_after
                    }
                    print(f"    ✅ Success: {len(results)} results")
                except Exception as e:
                    recovery_results[name] = {
                        'status': 'failed',
                        'error': str(e),
                        'before': stats_before
                    }
                    print(f"    ❌ Failed: {e}")
            else:
                print(f"  ⏭️  Skipping {name} (no recovery needed)")
                recovery_results[name] = {
                    'status': 'skipped',
                    'before': stats_before
                }
        
        processor.close()
        
        # Summary
        print(f"\n📊 Batch Recovery Summary:")
        for name, result in recovery_results.items():
            status = result['status']
            if status == 'success':
                before_rate = result['before']['completion_rate']
                after_rate = result['after']['completion_rate']
                improvement = after_rate - before_rate
                print(f"  ✅ {name}: {before_rate:.1f}% → {after_rate:.1f}% (+{improvement:.1f}%)")
            elif status == 'failed':
                print(f"  ❌ {name}: Failed - {result['error']}")
            else:
                print(f"  ⏭️  {name}: Skipped - no recovery needed")
        
    finally:
        # Cleanup
        for db_path in db_files:
            if os.path.exists(db_path):
                os.unlink(db_path)
        os.rmdir(temp_dir)


def example_4_error_handling():
    """Example 4: Comprehensive error handling during recovery."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    def safe_recovery(db_path: str, description: str) -> Optional[List[str]]:
        """Safely attempt recovery with comprehensive error handling."""
        print(f"\n🔄 Testing: {description}")
        
        processor = ParallelLLMProcessor(
            chat_fn=improved_chat_fn,
            num_workers=2
        )
        
        try:
            results = processor.recover_from_database(db_path)
            print(f"  ✅ Success: {len(results)} results recovered")
            return results
            
        except FileNotFoundError:
            print(f"  ❌ Database file not found: {db_path}")
            return None
            
        except ValueError as e:
            print(f"  ❌ Invalid database format: {e}")
            return None
            
        except sqlite3.Error as e:
            print(f"  ❌ Database error: {e}")
            return None
            
        except RuntimeError as e:
            print(f"  ❌ Recovery processing error: {e}")
            return None
            
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return None
            
        finally:
            processor.close()
    
    # Test 1: Non-existent database
    safe_recovery("nonexistent.db", "Non-existent database file")
    
    # Test 2: Invalid database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        invalid_db = tmp_file.name
        tmp_file.write(b"This is not a valid SQLite database")
    
    try:
        safe_recovery(invalid_db, "Invalid database file")
    finally:
        os.unlink(invalid_db)
    
    # Test 3: Database without required table
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        no_table_db = tmp_file.name
    
    try:
        conn = sqlite3.connect(no_table_db)
        conn.execute("CREATE TABLE wrong_table (id INTEGER)")
        conn.close()
        
        safe_recovery(no_table_db, "Database without llm_results table")
    finally:
        os.unlink(no_table_db)
    
    # Test 4: Valid database with successful recovery
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        valid_db = tmp_file.name
    
    try:
        create_test_database(valid_db, "few_failed")
        results = safe_recovery(valid_db, "Valid database with incomplete results")
        if results:
            print(f"    Sample result: {results[0]}")
    finally:
        os.unlink(valid_db)


def example_5_performance_optimization():
    """Example 5: Performance optimization strategies."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Performance Optimization")
    print("="*60)
    
    def timed_recovery(db_path: str, config: Dict, description: str) -> Tuple[float, int]:
        """Perform recovery and measure performance."""
        print(f"\n⏱️  Testing: {description}")
        
        start_time = time.time()
        
        with ParallelLLMProcessor(
            chat_fn=mock_chat_fn,
            **config
        ) as processor:
            results = processor.recover_from_database(db_path)
        
        duration = time.time() - start_time
        print(f"  ⏱️  Duration: {duration:.2f} seconds")
        print(f"  📊 Results: {len(results)}")
        print(f"  🚀 Rate: {len(results)/duration:.1f} results/second")
        
        return duration, len(results)
    
    # Create a larger test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create larger dataset
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE llm_results (
                id INTEGER PRIMARY KEY,
                prompt TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert 20 records with half incomplete
        test_data = []
        for i in range(1, 21):
            prompt = f"Test prompt {i}"
            result = f"Complete result {i}" if i % 2 == 0 else None
            test_data.append((i, prompt, result))
        
        cursor.executemany(
            "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
            test_data
        )
        
        conn.commit()
        conn.close()
        
        display_database_state(db_path, "Performance Test Database")
        
        # Test different configurations
        configs = [
            {
                'num_workers': 1,
                'retry_attempts': 1,
                'timeout': 30.0,
                'description': 'Conservative (1 worker, minimal retries)'
            },
            {
                'num_workers': 4,
                'retry_attempts': 2,
                'timeout': 30.0,
                'description': 'Balanced (4 workers, moderate retries)'
            },
            {
                'num_workers': 8,
                'retry_attempts': 1,
                'timeout': 15.0,
                'description': 'Aggressive (8 workers, fast timeout)'
            }
        ]
        
        results = []
        for config in configs:
            description = config.pop('description')
            duration, count = timed_recovery(db_path, config, description)
            results.append((description, duration, count))
        
        # Performance summary
        print(f"\n📊 Performance Comparison:")
        for description, duration, count in results:
            rate = count / duration if duration > 0 else 0
            print(f"  {description}: {duration:.2f}s ({rate:.1f} results/sec)")
        
        # Find best configuration
        best = min(results, key=lambda x: x[1])
        print(f"\n🏆 Best Performance: {best[0]} ({best[1]:.2f}s)")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def example_6_production_patterns():
    """Example 6: Production-ready recovery patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Production Patterns")
    print("="*60)
    
    class ProductionRecoveryManager:
        """Production-ready recovery management with monitoring and backup."""
        
        def __init__(self, llm_function, base_config=None):
            self.llm_function = llm_function
            self.base_config = base_config or {
                'num_workers': 4,
                'retry_attempts': 3,
                'timeout': 60.0
            }
            self.recovery_log = []
        
        def create_backup(self, db_path: str) -> str:
            """Create backup before recovery."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_path}.backup_{timestamp}"
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        
        def recover_with_backup(self, db_path: str) -> Optional[List[str]]:
            """Recover with automatic backup creation."""
            backup_path = self.create_backup(db_path)
            
            try:
                processor = ParallelLLMProcessor(
                    chat_fn=self.llm_function,
                    **self.base_config
                )
                
                results = processor.recover_from_database(db_path)
                processor.close()
                
                self.recovery_log.append({
                    'file': db_path,
                    'status': 'success',
                    'results_count': len(results),
                    'backup': backup_path,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"Recovery successful: {len(results)} results")
                return results
                
            except Exception as e:
                self.recovery_log.append({
                    'file': db_path,
                    'status': 'failed',
                    'error': str(e),
                    'backup': backup_path,
                    'timestamp': datetime.now()
                })
                
                logger.error(f"Recovery failed: {e}")
                logger.info(f"Original database preserved in backup: {backup_path}")
                raise
        
        def validate_recovery(self, db_path: str) -> bool:
            """Validate that recovery was successful."""
            stats = analyze_database_state(db_path)
            if not stats:
                return False
            
            success = stats['incomplete'] == 0
            logger.info(f"Recovery validation: {'PASSED' if success else 'FAILED'}")
            logger.info(f"  Completion rate: {stats['completion_rate']:.1f}%")
            
            return success
        
        def get_recovery_report(self) -> str:
            """Generate recovery report."""
            if not self.recovery_log:
                return "No recovery operations performed."
            
            report = ["Recovery Report", "=" * 15]
            
            for entry in self.recovery_log:
                report.append(f"\nFile: {entry['file']}")
                report.append(f"Status: {entry['status']}")
                report.append(f"Timestamp: {entry['timestamp']}")
                
                if entry['status'] == 'success':
                    report.append(f"Results: {entry['results_count']}")
                else:
                    report.append(f"Error: {entry['error']}")
                
                report.append(f"Backup: {entry['backup']}")
            
            return "\n".join(report)
    
    # Demonstrate production recovery
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test databases
        db_files = []
        for i in range(3):
            db_path = os.path.join(temp_dir, f"production_session_{i+1}.db")
            create_test_database(db_path, "mixed")
            db_files.append(db_path)
        
        print("📊 Created production test databases")
        
        # Initialize recovery manager
        manager = ProductionRecoveryManager(
            llm_function=improved_chat_fn,
            base_config={
                'num_workers': 3,
                'retry_attempts': 2,
                'timeout': 30.0
            }
        )
        
        # Perform recovery on all databases
        print("\n🔄 Performing production recovery...")
        
        for db_path in db_files:
            name = os.path.basename(db_path)
            print(f"\n  Processing: {name}")
            
            try:
                # Show before state
                stats_before = analyze_database_state(db_path)
                print(f"    Before: {stats_before['incomplete']} incomplete")
                
                # Perform recovery
                results = manager.recover_with_backup(db_path)
                
                # Validate recovery
                if manager.validate_recovery(db_path):
                    print(f"    ✅ Success: {len(results)} results, validation passed")
                else:
                    print(f"    ⚠️  Warning: Recovery completed but validation failed")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        # Generate and display report
        print(f"\n📋 Final Report:")
        print(manager.get_recovery_report())
        
    finally:
        # Cleanup (but keep backups in real production)
        for file in glob.glob(os.path.join(temp_dir, "*")):
            os.unlink(file)
        os.rmdir(temp_dir)


def main():
    """Run all recovery examples."""
    print("🔄 Comprehensive Recovery Examples for ParallelLLMProcessor")
    print("=" * 70)
    print("This demonstration covers various recovery scenarios and best practices.")
    
    try:
        example_1_basic_recovery()
        example_2_recovery_with_different_config()
        example_3_batch_recovery()
        example_4_error_handling()
        example_5_performance_optimization()
        example_6_production_patterns()
        
        print("\n" + "="*70)
        print("🎉 All recovery examples completed successfully!")
        print("\nKey Takeaways:")
        print("  • Recovery only reprocesses incomplete results")
        print("  • Complete results are preserved exactly")
        print("  • Recovery is idempotent and safe to repeat")
        print("  • Different configurations can improve recovery success")
        print("  • Production systems should include backup and validation")
        print("  • Error handling is crucial for robust recovery")
        print("  • Performance can be optimized based on workload")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()