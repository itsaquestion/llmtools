# llmtools

A Python library for parallel LLM inference processing with optional SQLite database storage.

## Features

- **Parallel Processing**: Concurrent execution of multiple LLM prompts while maintaining result order
- **Database Storage**: Optional real-time SQLite database storage for prompts and results
- **Thread Safety**: Safe concurrent database operations across multiple worker threads
- **Error Resilience**: Robust error handling with configurable retry mechanisms
- **Resource Management**: Automatic cleanup with context manager support
- **Performance Optimized**: Minimal overhead when database storage is enabled

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage (No Database)

```python
from src.llmtools import ParallelLLMProcessor

def my_llm_function(prompt: str) -> str:
    # Your LLM API call here
    return f"Response to: {prompt}"

# Process prompts in parallel
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4
)

prompts = ["Hello", "How are you?", "What is AI?"]
results = processor.process_prompts(prompts)
processor.close()

print(results)
# ['Response to: Hello', 'Response to: How are you?', 'Response to: What is AI?']
```

### With Database Storage

```python
from src.llmtools import ParallelLLMProcessor

# Enable database storage with custom filename
with ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=4,
    save_to_db=True,
    db_filename="my_llm_results.db"
) as processor:
    results = processor.process_prompts(prompts)

# Database automatically contains:
# - All input prompts
# - All results (including errors)
# - Timestamps for each record
```

### Using Default Database Filename

```python
# Automatically generates filename like "llm_results_1642694400.db"
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    save_to_db=True  # Uses timestamp-based filename
)
results = processor.process_prompts(prompts)
processor.close()
```

## Database Schema

When `save_to_db=True`, the processor creates a SQLite database with the following structure:

```sql
CREATE TABLE llm_results (
    id INTEGER PRIMARY KEY,           -- Corresponds to prompt index + 1
    prompt TEXT NOT NULL,             -- Input prompt text
    result TEXT,                      -- LLM response or error message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Record creation time
);
```

### Database Workflow

1. **Initialization**: All prompts are inserted with `result=NULL`
2. **Processing**: Results are updated in real-time as they complete
3. **Error Handling**: Failed prompts store error messages in the result field
4. **Order Preservation**: Database IDs correspond to original prompt order

## Configuration Options

### ParallelLLMProcessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_fn` | `Callable[[str], str]` | Required | LLM function that processes a single prompt |
| `num_workers` | `int` | `4` | Number of parallel worker threads |
| `retry_attempts` | `int` | `3` | Number of retry attempts for failed requests |
| `retry_delay` | `float` | `1.0` | Delay between retries in seconds |
| `timeout` | `float` | `60.0` | Timeout for each request in seconds |
| `save_to_db` | `bool` | `False` | Enable SQLite database storage |
| `db_filename` | `Optional[str]` | `None` | Custom database filename |

### Best Practices

#### Performance Optimization

```python
# For high-throughput processing
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=8,          # Increase workers for I/O bound tasks
    timeout=30.0,           # Shorter timeout for faster APIs
    retry_attempts=1,       # Reduce retries for stable APIs
    save_to_db=True
)
```

#### Error Handling

```python
# Robust configuration for unreliable APIs
processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    num_workers=2,          # Lower concurrency for unstable APIs
    retry_attempts=5,       # More retries for flaky connections
    retry_delay=2.0,        # Longer delay between retries
    timeout=120.0,          # Longer timeout for slow APIs
    save_to_db=True
)
```

#### Resource Management

```python
# Always use context managers for automatic cleanup
with ParallelLLMProcessor(
    chat_fn=my_llm_function,
    save_to_db=True,
    db_filename="important_results.db"
) as processor:
    results = processor.process_prompts(large_prompt_list)
    # Database connections automatically closed
```

## Advanced Usage

### Custom Error Handling

```python
def robust_llm_function(prompt: str) -> str:
    try:
        # Your LLM API call
        return call_llm_api(prompt)
    except APIRateLimitError:
        time.sleep(5)  # Wait for rate limit reset
        return call_llm_api(prompt)
    except APIError as e:
        return f"API Error: {str(e)}"

processor = ParallelLLMProcessor(
    chat_fn=robust_llm_function,
    save_to_db=True
)
```

### Monitoring Progress

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

processor = ParallelLLMProcessor(
    chat_fn=my_llm_function,
    save_to_db=True
)

# Progress bar is automatically shown during processing
results = processor.process_prompts(prompts)
```

### Database Analysis

```python
import sqlite3

# Analyze results after processing
conn = sqlite3.connect("my_llm_results.db")
cursor = conn.cursor()

# Count successful vs failed results
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN result LIKE 'Error:%' THEN 1 ELSE 0 END) as errors,
        SUM(CASE WHEN result NOT LIKE 'Error:%' THEN 1 ELSE 0 END) as success
    FROM llm_results
""")

total, errors, success = cursor.fetchone()
print(f"Total: {total}, Success: {success}, Errors: {errors}")

conn.close()
```

## Error Handling

The library provides robust error handling at multiple levels:

### Database Errors
- Database connection failures don't interrupt processing
- Invalid database paths gracefully degrade to non-database mode
- Database lock conflicts are automatically retried

### Processing Errors
- Individual prompt failures don't stop batch processing
- Configurable retry mechanism with exponential backoff
- Timeout protection prevents hanging on slow requests

### Resource Errors
- Automatic cleanup of database connections
- Context manager support for guaranteed resource cleanup
- Thread-safe resource management

## Performance Considerations

### Database Overhead
- Typical overhead: < 10% when database storage is enabled
- WAL mode enabled for better concurrent performance
- Batch initialization minimizes database operations

### Concurrency Tuning
- Start with `num_workers = 4` and adjust based on your API limits
- Higher worker counts benefit I/O bound operations
- Monitor API rate limits to avoid throttling

### Memory Usage
- Results are stored in memory until processing completes
- For very large batches, consider processing in chunks
- Database storage doesn't significantly increase memory usage

## Examples

See the `demo_resource_management.py` file for comprehensive examples including:
- Basic database functionality
- Performance comparisons
- Error resilience testing
- Concurrent safety validation
- Resource management patterns

## Requirements

- Python 3.7+
- SQLite3 (included with Python)
- tqdm (for progress bars)

## License

MIT License