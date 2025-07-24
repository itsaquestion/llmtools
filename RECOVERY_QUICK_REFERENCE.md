# Recovery Quick Reference

## Basic Recovery

```python
from src.llmtools import ParallelLLMProcessor

# Simple recovery
processor = ParallelLLMProcessor(chat_fn=my_llm_function)
results = processor.recover_from_database("session.db")
processor.close()
```

## Recovery with Context Manager (Recommended)

```python
with ParallelLLMProcessor(chat_fn=my_llm_function) as processor:
    results = processor.recover_from_database("session.db")
```

## Recovery with Custom Configuration

```python
with ParallelLLMProcessor(
    chat_fn=improved_llm_function,  # Better function
    num_workers=8,                  # More workers
    retry_attempts=5,               # More retries
    timeout=120.0                   # Longer timeout
) as processor:
    results = processor.recover_from_database("session.db")
```

## Error Handling

```python
try:
    results = processor.recover_from_database("session.db")
except FileNotFoundError:
    print("Database file not found")
except ValueError:
    print("Invalid database format")
except RuntimeError:
    print("Recovery processing failed")
```

## Check Database Before Recovery

```python
import sqlite3

def check_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM llm_results")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM llm_results 
        WHERE result IS NULL OR result = '' OR result = 'NA'
    """)
    incomplete = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Total: {total}, Incomplete: {incomplete}")
    return incomplete > 0

if check_database("session.db"):
    # Proceed with recovery
    pass
```

## Batch Recovery

```python
import glob

processor = ParallelLLMProcessor(chat_fn=my_llm_function)

for db_file in glob.glob("session_*.db"):
    try:
        results = processor.recover_from_database(db_file)
        print(f"✅ {db_file}: {len(results)} results")
    except Exception as e:
        print(f"❌ {db_file}: {e}")

processor.close()
```

## Key Points

- **Selective**: Only reprocesses incomplete results (NULL, "", "NA")
- **Preserves**: Existing successful results are never changed
- **Idempotent**: Safe to run multiple times
- **Ordered**: Results returned in original order
- **Configurable**: Uses current processor settings

## Common Patterns

### Production Recovery with Backup
```python
import shutil
from datetime import datetime

# Create backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = f"session.db.backup_{timestamp}"
shutil.copy2("session.db", backup)

# Perform recovery
with ParallelLLMProcessor(chat_fn=my_llm_function) as processor:
    results = processor.recover_from_database("session.db")
```

### Recovery with Monitoring
```python
import logging

logging.basicConfig(level=logging.INFO)

with ParallelLLMProcessor(chat_fn=my_llm_function) as processor:
    results = processor.recover_from_database("session.db")
    # Detailed progress will be logged automatically
```

### Robust LLM Function for Recovery
```python
def robust_llm_function(prompt: str) -> str:
    try:
        return original_llm_function(prompt)
    except Exception as e:
        return f"Error: {str(e)}"  # Return error instead of failing

processor = ParallelLLMProcessor(chat_fn=robust_llm_function)
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Database file not found" | Check file path exists |
| "Invalid database format" | Verify SQLite file with llm_results table |
| "Database is locked" | Close other connections, wait and retry |
| "Recovery processing error" | Check LLM function, API credentials |
| Some results still incomplete | Run recovery again, improve LLM function |

## Performance Tips

- **Small datasets**: `num_workers=2`, `timeout=15.0`
- **Large datasets**: `num_workers=6`, `timeout=60.0`
- **Rate limits**: `num_workers=1`, add delays in LLM function
- **Memory constraints**: Use fewer workers, process in batches