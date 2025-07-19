# Database Usage Guide for ParallelLLMProcessor

This guide provides comprehensive information about using the SQLite database storage feature in ParallelLLMProcessor, including best practices, troubleshooting, and advanced usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [Basic Usage](#basic-usage)
3. [Database Schema](#database-schema)
4. [Configuration Options](#configuration-options)
5. [Best Practices](#best-practices)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

The database storage feature allows you to persist all prompts and results from your LLM processing sessions. This is useful for:

- **Data Analysis**: Analyze patterns in prompts and responses
- **Debugging**: Track down issues with specific prompts
- **Auditing**: Maintain records of all LLM interactions
- **Resuming**: Continue processing from where you left off
- **Monitoring**: Track success rates and performance metrics

### Key Features

- **Real-time Storage**: Results are saved immediately upon completion
- **Thread-safe Operations**: Multiple workers can safely update the database
- **Error Resilience**: Database failures don't interrupt processing
- **Order Preservation**: Database records maintain original prompt order
- **Automatic Cleanup**: Context managers ensure proper resource management

## Basic Usage

### Enable Database Storage

```python
from src.llmtools import ParallelLLMProcessor

# Basic database usage
with ParallelLLMProcessor(
    chat_fn=your_llm_function,
    save_to_db=True,
    db_filename="my_results.db"
) as processor:
    results = processor.process_prompts(prompts)
```

### Using Default Filename

```python
# Automatically generates timestamp-based filename
processor = ParallelLLMProcessor(
    chat_fn=your_llm_function,
    save_to_db=True  # Creates llm_results_{YYYYMMDD_HHMMSS}.db
)
```

## Database Schema

The database contains a single table `llm_results` with the following structure:

```sql
CREATE TABLE llm_results (
    id INTEGER PRIMARY KEY,           -- Prompt index + 1
    prompt TEXT NOT NULL,             -- Original prompt text
    result TEXT,                      -- LLM response or error message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Field Descriptions

- **id**: Sequential identifier corresponding to prompt position in original list (1-based)
- **prompt**: The exact prompt text sent to the LLM
- **result**: The LLM response, or error message if processing failed
- **created_at**: Timestamp when the record was initially created

### Data Flow

1. **Initialization**: All prompts inserted with `result = NULL`
2. **Processing**: Results updated in real-time as they complete
3. **Completion**: All records have either successful results or error messages

## Configuration Options

### Database Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_to_db` | `bool` | `False` | Enable database storage |
| `db_filename` | `Optional[str]` | `None` | Custom database filename |

### Related Parameters

| Parameter | Type | Default | Impact on Database |
|-----------|------|---------|-------------------|
| `num_workers` | `int` | `4` | Higher concurrency = more database contention |
| `retry_attempts` | `int` | `3` | Failed retries still update database with errors |
| `timeout` | `float` | `60.0` | Timeouts result in error records |

## Best Practices

### 1. Use Context Managers

Always use context managers to ensure proper database cleanup:

```python
# ✅ Good - automatic cleanup
with ParallelLLMProcessor(save_to_db=True, db_filename="results.db") as processor:
    results = processor.process_prompts(prompts)

# ❌ Avoid - manual cleanup required
processor = ParallelLLMProcessor(save_to_db=True, db_filename="results.db")
results = processor.process_prompts(prompts)
processor.close()  # Must remember to call this
```

### 2. Choose Appropriate Filenames

```python
# ✅ Good - descriptive filename
db_filename = f"experiment_results_{experiment_id}_{timestamp}.db"

# ✅ Good - use default for quick tests
save_to_db=True  # Auto-generates unique filename

# ❌ Avoid - generic names that might conflict
db_filename = "results.db"
```

### 3. Handle Large Batches

For very large prompt batches, consider processing in chunks:

```python
def process_large_batch(prompts, chunk_size=1000):
    all_results = []
    
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i:i + chunk_size]
        chunk_filename = f"results_chunk_{i//chunk_size}.db"
        
        with ParallelLLMProcessor(
            chat_fn=llm_function,
            save_to_db=True,
            db_filename=chunk_filename
        ) as processor:
            chunk_results = processor.process_prompts(chunk)
            all_results.extend(chunk_results)
    
    return all_results
```

### 4. Configure for Your Use Case

```python
# High-throughput, stable API
processor = ParallelLLMProcessor(
    chat_fn=llm_function,
    num_workers=8,
    retry_attempts=1,
    timeout=30.0,
    save_to_db=True
)

# Unreliable API, need robustness
processor = ParallelLLMProcessor(
    chat_fn=llm_function,
    num_workers=2,
    retry_attempts=5,
    retry_delay=2.0,
    timeout=120.0,
    save_to_db=True
)
```

## Error Handling

### Database Error Resilience

The system is designed to continue processing even when database operations fail:

```python
# Database errors are logged but don't stop processing
processor = ParallelLLMProcessor(
    chat_fn=llm_function,
    save_to_db=True,
    db_filename="/invalid/path/results.db"  # Will fail
)

# Processing continues, database functionality disabled
results = processor.process_prompts(prompts)  # Still works!
```

### Common Error Scenarios

1. **Invalid Database Path**: Processing continues without database
2. **Database Locked**: Automatic retry with exponential backoff
3. **Disk Full**: Error logged, processing continues
4. **Permission Denied**: Database disabled, processing continues

### Error Logging

Enable logging to monitor database operations:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Database operations will be logged
processor = ParallelLLMProcessor(save_to_db=True)
```

## Performance Considerations

### Database Overhead

Typical performance impact:
- **Overhead**: < 10% in most cases
- **Bottleneck**: Usually the LLM API, not database operations
- **Scaling**: Linear with number of prompts

### Optimization Tips

1. **Use WAL Mode**: Automatically enabled for better concurrency
2. **Batch Operations**: Initialization uses batch inserts
3. **Connection Pooling**: Single connection per processor instance
4. **Lock Minimization**: Fine-grained locking for updates

### Performance Testing

```python
import time

# Measure overhead
start = time.time()
processor_no_db = ParallelLLMProcessor(save_to_db=False)
results_no_db = processor_no_db.process_prompts(test_prompts)
time_no_db = time.time() - start

start = time.time()
processor_with_db = ParallelLLMProcessor(save_to_db=True)
results_with_db = processor_with_db.process_prompts(test_prompts)
time_with_db = time.time() - start

overhead = (time_with_db - time_no_db) / time_no_db * 100
print(f"Database overhead: {overhead:.1f}%")
```

## Advanced Usage

### Database Analysis

```python
import sqlite3
import pandas as pd

# Load results into pandas for analysis
conn = sqlite3.connect("results.db")
df = pd.read_sql_query("SELECT * FROM llm_results", conn)

# Analyze success rates
success_rate = (df['result'].str.startswith('Error:') == False).mean()
print(f"Success rate: {success_rate:.1%}")

# Find longest responses
longest_responses = df.nlargest(5, df['result'].str.len())
print(longest_responses[['prompt', 'result']])

conn.close()
```

### Custom Database Queries

```python
def analyze_results(db_filename):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Success/failure statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN result LIKE 'Error:%' THEN 1 ELSE 0 END) as errors,
            AVG(LENGTH(result)) as avg_response_length
        FROM llm_results
    """)
    
    stats = cursor.fetchone()
    print(f"Total: {stats[0]}, Errors: {stats[1]}, Avg Length: {stats[2]:.1f}")
    
    # Response time analysis (if you add timing)
    cursor.execute("""
        SELECT prompt, result, created_at 
        FROM llm_results 
        ORDER BY created_at
    """)
    
    conn.close()
```

### Resuming Interrupted Processing

```python
def resume_processing(db_filename, all_prompts):
    """Resume processing from where it left off."""
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Find completed prompts
    cursor.execute("SELECT id FROM llm_results WHERE result IS NOT NULL")
    completed_ids = {row[0] for row in cursor.fetchall()}
    
    # Find remaining prompts (convert to 0-based indexing)
    remaining_indices = [i for i in range(len(all_prompts)) 
                        if (i + 1) not in completed_ids]
    remaining_prompts = [all_prompts[i] for i in remaining_indices]
    
    conn.close()
    
    if remaining_prompts:
        print(f"Resuming processing of {len(remaining_prompts)} remaining prompts")
        # Process remaining prompts...
    else:
        print("All prompts already processed")
```

## Troubleshooting

### Common Issues

#### 1. Database File Not Created

**Problem**: `save_to_db=True` but no database file appears

**Solutions**:
- Check that `process_prompts()` was called with non-empty prompt list
- Verify write permissions in the target directory
- Check logs for database initialization errors

#### 2. Database Locked Errors

**Problem**: "Database is locked" errors in logs

**Solutions**:
- Reduce `num_workers` to decrease contention
- Ensure no other processes are accessing the database
- Check disk space and permissions

#### 3. Missing Results in Database

**Problem**: Some results are NULL in database

**Solutions**:
- Check if processing was interrupted
- Look for error messages in logs
- Verify all prompts completed processing

#### 4. Performance Issues

**Problem**: Database storage significantly slows processing

**Solutions**:
- Check disk I/O performance
- Reduce `num_workers` if database contention is high
- Consider using SSD storage for database files

### Debugging Tips

1. **Enable Detailed Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Database Contents**:
   ```python
   import sqlite3
   conn = sqlite3.connect("your_db.db")
   cursor = conn.cursor()
   cursor.execute("SELECT COUNT(*) FROM llm_results")
   print(f"Total records: {cursor.fetchone()[0]}")
   ```

3. **Monitor File System**:
   ```bash
   # Check database file size
   ls -lh *.db
   
   # Monitor disk space
   df -h .
   ```

## Examples

See `examples_database_usage.py` for comprehensive examples including:

- Basic database functionality
- Error handling scenarios
- Performance comparisons
- Database analysis techniques
- Best practices demonstrations

### Quick Reference

```python
# Basic usage
with ParallelLLMProcessor(save_to_db=True, db_filename="results.db") as p:
    results = p.process_prompts(prompts)

# Default filename
processor = ParallelLLMProcessor(save_to_db=True)

# High performance
processor = ParallelLLMProcessor(
    save_to_db=True, num_workers=8, timeout=30.0
)

# High reliability
processor = ParallelLLMProcessor(
    save_to_db=True, retry_attempts=5, retry_delay=2.0
)
```

## Support

For additional help:
1. Check the main README.md for general usage
2. Run `examples_database_usage.py` for working examples
3. Enable debug logging to diagnose issues
4. Review the source code for detailed implementation