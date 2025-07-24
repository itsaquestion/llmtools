# Recovery Functionality Usage Guide

This guide provides comprehensive documentation for the `recover_from_database` method in ParallelLLMProcessor, which allows you to resume interrupted processing sessions and recover incomplete results.

## Overview

The recovery functionality enables you to:
- Resume processing from an interrupted session
- Reprocess failed or incomplete results
- Maintain data integrity and result order
- Avoid losing previous work due to interruptions

## When to Use Recovery

Recovery is useful in these scenarios:
- **Interrupted Processing**: Your script was stopped before completion
- **Partial Failures**: Some prompts failed due to API issues or timeouts
- **System Crashes**: Processing was interrupted by system issues
- **Network Issues**: Temporary connectivity problems caused failures
- **API Rate Limits**: Processing was stopped due to rate limiting

## How Recovery Works

1. **Database Analysis**: Scans the existing database for incomplete results
2. **Identification**: Finds records with NULL, empty string, or "NA" results
3. **Reprocessing**: Uses current processor configuration to reprocess failed prompts
4. **Database Update**: Updates the database with new results
5. **Result Return**: Returns complete results in original order

## Basic Usage

### Simple Recovery Example

```python
from src.llmtools import ParallelLLMProcessor

def my_llm_function(prompt: str) -> str:
    # Your LLM API call here
    return f"Response to: {prompt}"

# Create processor with same or updated configuration
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4,
    retry_attempts=3
)

# Recover from existing database
results = processor.recover_from_database("previous_session.db")
processor.close()

print(f"Recovered {len(results)} results")
```

### Recovery with Context Manager

```python
# Recommended approach using context manager
with ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4
) as processor:
    results = processor.recover_from_database("interrupted_session.db")
    
# Database connections automatically cleaned up
```

## Advanced Usage Patterns

### Recovery with Different Configuration

You can recover using different processor settings than the original session:

```python
# Original session might have used different settings
# Recovery can use optimized configuration
processor = ParallelLLMProcessor(
    chat_fn=improved_llm_function,  # Updated function
    num_workers=8,                  # More workers
    retry_attempts=5,               # More retries
    timeout=120.0                   # Longer timeout
)

results = processor.recover_from_database("original_session.db")
```

### Batch Recovery for Multiple Databases

```python
import glob
from pathlib import Path

def recover_all_incomplete_sessions(pattern: str):
    """Recover from all database files matching a pattern."""
    db_files = glob.glob(pattern)
    
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        num_workers=4
    )
    
    recovered_sessions = {}
    
    for db_file in db_files:
        try:
            print(f"Recovering from {db_file}...")
            results = processor.recover_from_database(db_file)
            recovered_sessions[db_file] = results
            print(f"✓ Recovered {len(results)} results from {db_file}")
        except Exception as e:
            print(f"✗ Failed to recover from {db_file}: {e}")
    
    processor.close()
    return recovered_sessions

# Recover from all databases in current directory
sessions = recover_all_incomplete_sessions("llm_results_*.db")
```

### Recovery with Progress Monitoring

```python
import logging

# Enable detailed logging to monitor recovery progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4
)

# Recovery will show detailed progress information
results = processor.recover_from_database("large_session.db")
```

## Error Handling

### Handling Recovery Errors

```python
from src.llmtools import ParallelLLMProcessor
import sqlite3

def safe_recovery(db_path: str):
    """Safely attempt recovery with comprehensive error handling."""
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        num_workers=4
    )
    
    try:
        results = processor.recover_from_database(db_path)
        print(f"✓ Successfully recovered {len(results)} results")
        return results
        
    except FileNotFoundError:
        print(f"✗ Database file not found: {db_path}")
        return None
        
    except ValueError as e:
        print(f"✗ Invalid database format: {e}")
        return None
        
    except sqlite3.Error as e:
        print(f"✗ Database error: {e}")
        return None
        
    except RuntimeError as e:
        print(f"✗ Recovery processing error: {e}")
        return None
        
    finally:
        processor.close()

# Use safe recovery
results = safe_recovery("potentially_corrupted.db")
if results:
    print("Recovery successful!")
else:
    print("Recovery failed - check error messages above")
```

### Partial Recovery Handling

```python
def recovery_with_validation(db_path: str):
    """Perform recovery and validate results."""
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        num_workers=4
    )
    
    try:
        results = processor.recover_from_database(db_path)
        
        # Validate recovery completeness
        incomplete_count = sum(1 for r in results if not r or r in ["", "NA"])
        
        if incomplete_count == 0:
            print(f"✓ Complete recovery: all {len(results)} results recovered")
        else:
            print(f"⚠ Partial recovery: {incomplete_count} results still incomplete")
            print("Consider running recovery again or checking your LLM function")
        
        return results
        
    finally:
        processor.close()

results = recovery_with_validation("session.db")
```

## Database Analysis Before Recovery

### Checking Database State

```python
import sqlite3

def analyze_database_before_recovery(db_path: str):
    """Analyze database state before attempting recovery."""
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
            WHERE result LIKE 'Error:%'
        """)
        error_records = cursor.fetchone()[0]
        
        complete_records = total_records - incomplete_records
        
        print(f"Database Analysis: {db_path}")
        print(f"  Total records: {total_records}")
        print(f"  Complete records: {complete_records}")
        print(f"  Incomplete records: {incomplete_records}")
        print(f"  Error records: {error_records}")
        print(f"  Completion rate: {(complete_records/total_records*100):.1f}%")
        
        if incomplete_records > 0:
            print(f"  → Recovery recommended: {incomplete_records} records need reprocessing")
        else:
            print(f"  → No recovery needed: all records are complete")
        
        conn.close()
        return {
            'total': total_records,
            'complete': complete_records,
            'incomplete': incomplete_records,
            'errors': error_records
        }
        
    except Exception as e:
        print(f"Failed to analyze database: {e}")
        return None

# Analyze before recovery
stats = analyze_database_before_recovery("my_session.db")
if stats and stats['incomplete'] > 0:
    # Proceed with recovery
    processor = ParallelLLMProcessor(chat_fn=my_llm_function)
    results = processor.recover_from_database("my_session.db")
    processor.close()
```

## Performance Optimization

### Optimizing Recovery Performance

```python
def optimized_recovery(db_path: str, estimated_incomplete: int):
    """Recovery with performance optimizations based on workload."""
    
    # Adjust workers based on expected workload
    if estimated_incomplete < 10:
        num_workers = 2  # Small workload
    elif estimated_incomplete < 100:
        num_workers = 4  # Medium workload
    else:
        num_workers = 8  # Large workload
    
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        num_workers=num_workers,
        retry_attempts=2,  # Fewer retries for faster recovery
        timeout=30.0       # Shorter timeout for faster processing
    )
    
    results = processor.recover_from_database(db_path)
    processor.close()
    
    return results
```

### Memory-Efficient Recovery for Large Databases

```python
def memory_efficient_recovery(db_path: str):
    """Recovery approach for very large databases."""
    
    # Use fewer workers to reduce memory usage
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        num_workers=2,     # Reduced workers
        timeout=60.0       # Standard timeout
    )
    
    try:
        results = processor.recover_from_database(db_path)
        return results
    finally:
        processor.close()
```

## Integration Patterns

### Recovery in Production Workflows

```python
import os
import time
from datetime import datetime

class ProductionRecoveryManager:
    """Production-ready recovery management."""
    
    def __init__(self, llm_function, base_config=None):
        self.llm_function = llm_function
        self.base_config = base_config or {
            'num_workers': 4,
            'retry_attempts': 3,
            'timeout': 60.0
        }
    
    def recover_with_backup(self, db_path: str):
        """Recover with automatic backup creation."""
        # Create backup before recovery
        backup_path = f"{db_path}.backup_{int(time.time())}"
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Created backup: {backup_path}")
        
        try:
            processor = ParallelLLMProcessor(
                chat_fn=self.llm_function,
                **self.base_config
            )
            
            results = processor.recover_from_database(db_path)
            processor.close()
            
            print(f"Recovery successful, backup retained: {backup_path}")
            return results
            
        except Exception as e:
            print(f"Recovery failed: {e}")
            print(f"Original database preserved in backup: {backup_path}")
            raise
    
    def scheduled_recovery(self, db_pattern: str):
        """Perform scheduled recovery of incomplete sessions."""
        import glob
        
        db_files = glob.glob(db_pattern)
        recovery_log = []
        
        for db_file in db_files:
            try:
                # Check if recovery is needed
                stats = analyze_database_before_recovery(db_file)
                if stats and stats['incomplete'] > 0:
                    print(f"Starting recovery for {db_file}...")
                    results = self.recover_with_backup(db_file)
                    recovery_log.append({
                        'file': db_file,
                        'status': 'success',
                        'results_count': len(results),
                        'timestamp': datetime.now()
                    })
                else:
                    print(f"Skipping {db_file} - no recovery needed")
                    
            except Exception as e:
                recovery_log.append({
                    'file': db_file,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now()
                })
        
        return recovery_log

# Usage in production
manager = ProductionRecoveryManager(my_llm_function)
recovery_results = manager.scheduled_recovery("llm_results_*.db")
```

### Recovery with Monitoring and Alerts

```python
import smtplib
from email.mime.text import MIMEText

class MonitoredRecovery:
    """Recovery with monitoring and alerting."""
    
    def __init__(self, llm_function, alert_email=None):
        self.llm_function = llm_function
        self.alert_email = alert_email
    
    def send_alert(self, subject: str, message: str):
        """Send email alert (configure SMTP settings)."""
        if not self.alert_email:
            return
        
        # Configure your SMTP settings here
        # This is a placeholder implementation
        print(f"ALERT: {subject}")
        print(f"Message: {message}")
    
    def monitored_recovery(self, db_path: str):
        """Recovery with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            # Pre-recovery analysis
            stats = analyze_database_before_recovery(db_path)
            if not stats:
                raise ValueError("Could not analyze database")
            
            if stats['incomplete'] == 0:
                print("No recovery needed")
                return None
            
            # Perform recovery
            processor = ParallelLLMProcessor(
                chat_fn=self.llm_function,
                num_workers=4
            )
            
            results = processor.recover_from_database(db_path)
            processor.close()
            
            # Post-recovery analysis
            duration = time.time() - start_time
            
            # Send success notification
            self.send_alert(
                f"Recovery Successful: {db_path}",
                f"Recovered {len(results)} results in {duration:.2f} seconds"
            )
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Send failure alert
            self.send_alert(
                f"Recovery Failed: {db_path}",
                f"Error after {duration:.2f} seconds: {str(e)}"
            )
            
            raise

# Usage with monitoring
monitor = MonitoredRecovery(my_llm_function, "admin@company.com")
results = monitor.monitored_recovery("critical_session.db")
```

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good - automatic cleanup
with ParallelLLMProcessor(chat_fn=my_llm_function) as processor:
    results = processor.recover_from_database("session.db")

# ❌ Avoid - manual cleanup required
processor = ParallelLLMProcessor(chat_fn=my_llm_function)
results = processor.recover_from_database("session.db")
processor.close()  # Easy to forget
```

### 2. Validate Database Before Recovery

```python
# ✅ Good - check database state first
stats = analyze_database_before_recovery("session.db")
if stats and stats['incomplete'] > 0:
    # Proceed with recovery
    pass
else:
    print("No recovery needed")

# ❌ Avoid - blind recovery attempts
results = processor.recover_from_database("session.db")  # May be unnecessary
```

### 3. Handle Errors Gracefully

```python
# ✅ Good - comprehensive error handling
try:
    results = processor.recover_from_database("session.db")
except FileNotFoundError:
    print("Database file not found")
except ValueError as e:
    print(f"Invalid database: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# ❌ Avoid - no error handling
results = processor.recover_from_database("session.db")  # May crash
```

### 4. Use Appropriate Configuration

```python
# ✅ Good - configuration matches workload
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4,      # Reasonable for most APIs
    retry_attempts=3,   # Handle temporary failures
    timeout=60.0        # Prevent hanging
)

# ❌ Avoid - extreme configurations
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=50,     # May overwhelm API
    retry_attempts=0,   # No resilience
    timeout=1.0         # Too aggressive
)
```

### 5. Monitor Recovery Progress

```python
# ✅ Good - enable logging for visibility
import logging
logging.basicConfig(level=logging.INFO)

results = processor.recover_from_database("session.db")
# Will show detailed progress information

# ❌ Avoid - silent recovery
results = processor.recover_from_database("session.db")  # No visibility
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Database file not found"
```python
# Solution: Check file path and existence
import os
db_path = "my_session.db"
if not os.path.exists(db_path):
    print(f"Database file does not exist: {db_path}")
    # Check current directory or provide full path
```

#### Issue: "Invalid database format"
```python
# Solution: Verify database structure
import sqlite3
try:
    conn = sqlite3.connect("session.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables in database: {tables}")
    conn.close()
except Exception as e:
    print(f"Database verification failed: {e}")
```

#### Issue: Recovery takes too long
```python
# Solution: Optimize configuration
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=2,      # Reduce workers
    timeout=30.0,       # Shorter timeout
    retry_attempts=1    # Fewer retries
)
```

#### Issue: Some results still incomplete after recovery
```python
# Solution: Check LLM function and run recovery again
def robust_llm_function(prompt: str) -> str:
    try:
        return my_original_llm_function(prompt)
    except Exception as e:
        # Return error message instead of failing
        return f"Processing failed: {str(e)}"

# Recovery is idempotent - safe to run multiple times
results = processor.recover_from_database("session.db")
```

## Summary

The recovery functionality provides a robust way to resume interrupted processing sessions:

- **Automatic Detection**: Identifies incomplete results automatically
- **Selective Processing**: Only reprocesses failed or incomplete prompts
- **Data Integrity**: Preserves existing successful results
- **Order Preservation**: Maintains original result order
- **Idempotent**: Safe to run multiple times
- **Configurable**: Uses current processor settings for recovery

Use recovery whenever you need to resume interrupted processing or fix incomplete results from previous sessions.
## Fre
quently Asked Questions (FAQ)

### Q: When should I use recovery instead of reprocessing everything?

**A:** Use recovery when:
- You have a large dataset and only some results failed
- Processing is expensive (time/cost) and you want to preserve successful results
- You were interrupted mid-processing and want to resume
- You want to retry failed results with different configuration

Reprocess everything when:
- The dataset is small and reprocessing is quick
- You want to change the LLM function significantly
- You suspect the successful results might be incorrect

### Q: Can I use recovery with a different LLM function than the original?

**A:** Yes! Recovery uses the current processor configuration, so you can:
- Use an improved LLM function with better error handling
- Switch to a different LLM model or API
- Update your function with bug fixes or improvements

The recovery process will only reprocess incomplete results with your new function.

### Q: What happens if recovery fails partway through?

**A:** Recovery is designed to be resilient:
- Partial failures don't affect successful updates
- You can run recovery again - it's idempotent
- Failed updates are logged but don't stop the process
- The database remains in a consistent state

### Q: How do I know if recovery is working correctly?

**A:** Monitor the logs for:
- Number of incomplete records found
- Processing progress with progress bars
- Database update success/failure counts
- Final completion rate improvement

You can also analyze the database before and after recovery to verify results.

### Q: Can I run recovery multiple times on the same database?

**A:** Yes! Recovery is idempotent, meaning:
- Running it multiple times produces the same result
- It only processes records that are still incomplete
- Previously successful results are never reprocessed
- It's safe to run recovery again if some results still failed

### Q: What's the difference between incomplete and error results?

**A:** 
- **Incomplete results**: NULL, empty string (""), or "NA" values that need reprocessing
- **Error results**: Results that start with "Error:" or "Processing failed" - these are complete but indicate failures

Recovery reprocesses incomplete results but preserves error results (they're considered "complete" even if they represent failures).

### Q: How can I improve recovery success rates?

**A:** Try these strategies:
1. **Improve your LLM function**: Add better error handling and retry logic
2. **Adjust processor configuration**: More retries, longer timeouts, different worker counts
3. **Use a more reliable LLM service**: Switch to a more stable API
4. **Run recovery multiple times**: Some transient issues may resolve on retry

### Q: Can I recover from databases created by different processor versions?

**A:** Yes, as long as the database schema is compatible:
- The `llm_results` table structure must match
- Column names and types should be consistent
- Recovery works with any database that follows the expected format

### Q: What should I do if recovery is very slow?

**A:** Optimize performance by:
- Reducing `num_workers` if you're hitting API rate limits
- Decreasing `timeout` for faster APIs
- Reducing `retry_attempts` for stable APIs
- Using a faster LLM service or local model
- Processing in smaller batches if the database is very large

### Q: How do I handle databases with thousands of incomplete results?

**A:** For large-scale recovery:
- Use fewer workers (2-4) to avoid overwhelming APIs
- Enable detailed logging to monitor progress
- Consider processing in chunks if memory is limited
- Use a robust LLM function with good error handling
- Plan for longer processing times

### Q: Can I modify the database while recovery is running?

**A:** It's not recommended because:
- Recovery assumes the database structure remains stable
- Concurrent modifications could cause conflicts
- Results might be inconsistent

If you need to modify the database, stop recovery, make changes, then restart.

### Q: What happens to timestamps during recovery?

**A:** Recovery preserves original timestamps:
- The `created_at` field is not modified
- Only the `result` field is updated
- You can track when recovery occurred through logs

### Q: How do I backup my database before recovery?

**A:** Always backup before recovery:

```python
import shutil
from datetime import datetime

# Create timestamped backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"my_database.db.backup_{timestamp}"
shutil.copy2("my_database.db", backup_path)
print(f"Backup created: {backup_path}")

# Then proceed with recovery
processor = ParallelLLMProcessor(chat_fn=my_llm_function)
results = processor.recover_from_database("my_database.db")
processor.close()
```

## Common Error Messages and Solutions

### Error: "Database file not found"

**Cause**: The specified database file doesn't exist or path is incorrect.

**Solutions**:
- Check the file path is correct and file exists
- Use absolute paths to avoid directory confusion
- Verify file permissions allow reading

```python
import os
db_path = "my_session.db"
if os.path.exists(db_path):
    print("Database file exists")
else:
    print(f"Database file not found: {db_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
```

### Error: "Invalid database format"

**Cause**: The file exists but isn't a valid SQLite database or lacks required tables.

**Solutions**:
- Verify the file is a SQLite database
- Check that the `llm_results` table exists
- Ensure the table has the correct schema

```python
import sqlite3
try:
    conn = sqlite3.connect("my_session.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables in database: {tables}")
    
    if 'llm_results' in tables:
        cursor.execute("PRAGMA table_info(llm_results)")
        columns = cursor.fetchall()
        print(f"llm_results columns: {columns}")
    else:
        print("llm_results table not found")
    
    conn.close()
except Exception as e:
    print(f"Database verification failed: {e}")
```

### Error: "Recovery processing error"

**Cause**: Issues during the actual reprocessing of prompts.

**Solutions**:
- Check your LLM function for errors
- Verify API credentials and connectivity
- Reduce worker count to avoid rate limits
- Increase timeout for slow APIs

```python
def robust_llm_function(prompt: str) -> str:
    """LLM function with comprehensive error handling."""
    try:
        # Your original LLM call
        return original_llm_function(prompt)
    except APIRateLimitError:
        time.sleep(5)  # Wait for rate limit reset
        return original_llm_function(prompt)
    except APITimeoutError:
        return f"Timeout processing: {prompt[:50]}..."
    except Exception as e:
        return f"Error processing '{prompt[:30]}...': {str(e)}"
```

### Error: "Database is locked"

**Cause**: Another process is using the database or previous connection wasn't closed properly.

**Solutions**:
- Ensure no other processes are accessing the database
- Use context managers to guarantee connection cleanup
- Wait and retry if the lock is temporary

```python
import time
import sqlite3

def wait_for_database_unlock(db_path: str, max_attempts: int = 5):
    """Wait for database to become available."""
    for attempt in range(max_attempts):
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                print(f"Database locked, waiting... (attempt {attempt + 1})")
                time.sleep(2)
            else:
                raise
    return False

# Use before recovery
if wait_for_database_unlock("my_session.db"):
    # Proceed with recovery
    pass
else:
    print("Database remains locked, cannot proceed")
```

### Warning: "Some results still incomplete after recovery"

**Cause**: Some prompts failed even during recovery.

**Solutions**:
- Run recovery again (it's idempotent)
- Improve your LLM function's error handling
- Check for API issues or rate limits
- Consider manual inspection of failed prompts

```python
def analyze_remaining_failures(db_path: str):
    """Analyze what prompts are still failing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, prompt, result FROM llm_results 
        WHERE result IS NULL OR result = '' OR result = 'NA'
        LIMIT 10
    """)
    
    failures = cursor.fetchall()
    conn.close()
    
    print(f"Remaining failures ({len(failures)} shown):")
    for record_id, prompt, result in failures:
        print(f"  {record_id}: {prompt[:50]}... -> {repr(result)}")

# Analyze after recovery
analyze_remaining_failures("my_session.db")
```

## Performance Tuning Guide

### Optimizing Recovery Speed

**For Small Datasets (< 100 incomplete results)**:
```python
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=2,      # Lower concurrency
    retry_attempts=1,   # Fewer retries
    timeout=15.0        # Shorter timeout
)
```

**For Medium Datasets (100-1000 incomplete results)**:
```python
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4,      # Balanced concurrency
    retry_attempts=2,   # Moderate retries
    timeout=30.0        # Standard timeout
)
```

**For Large Datasets (> 1000 incomplete results)**:
```python
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=6,      # Higher concurrency
    retry_attempts=3,   # More retries for reliability
    timeout=60.0        # Longer timeout for stability
)
```

### Memory Optimization

For very large databases, consider:
- Using fewer workers to reduce memory usage
- Processing in batches if possible
- Monitoring system memory during recovery

### API Rate Limit Handling

If you're hitting API rate limits:
- Reduce `num_workers` to 1-2
- Increase delays in your LLM function
- Use exponential backoff in error handling
- Consider using multiple API keys with load balancing

## Integration with Monitoring Systems

### Logging Integration

```python
import logging
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recovery.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Recovery will automatically log detailed progress
processor = ParallelLLMProcessor(chat_fn=my_llm_function)
results = processor.recover_from_database("session.db")
```

### Metrics Collection

```python
import time
from datetime import datetime

def recovery_with_metrics(db_path: str):
    """Recovery with comprehensive metrics collection."""
    metrics = {
        'start_time': datetime.now(),
        'database_file': db_path,
        'processor_config': {
            'num_workers': 4,
            'retry_attempts': 3,
            'timeout': 60.0
        }
    }
    
    # Pre-recovery analysis
    stats_before = analyze_database_state(db_path)
    metrics['before_recovery'] = stats_before
    
    # Perform recovery
    start_time = time.time()
    
    processor = ParallelLLMProcessor(
        chat_fn=my_llm_function,
        **metrics['processor_config']
    )
    
    try:
        results = processor.recover_from_database(db_path)
        metrics['success'] = True
        metrics['results_count'] = len(results)
    except Exception as e:
        metrics['success'] = False
        metrics['error'] = str(e)
        raise
    finally:
        processor.close()
        metrics['duration'] = time.time() - start_time
        metrics['end_time'] = datetime.now()
    
    # Post-recovery analysis
    stats_after = analyze_database_state(db_path)
    metrics['after_recovery'] = stats_after
    
    # Calculate improvement
    if stats_before and stats_after:
        improvement = stats_after['completion_rate'] - stats_before['completion_rate']
        metrics['completion_improvement'] = improvement
    
    return results, metrics

# Use with metrics
results, metrics = recovery_with_metrics("session.db")
print(f"Recovery metrics: {metrics}")
```

This comprehensive guide should help users understand and effectively use the recovery functionality in all scenarios, from basic usage to production deployments.