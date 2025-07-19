import time

def mock_chat_fn(prompt: str) -> str:
    """Mock LLM function for testing"""
    time.sleep(0.5)  # Simulate API delay
    return f"Response to: {prompt}"

# Test prompts
test_prompts = [
    "1. Tell me a little story about AI.",
    "2. What color is a ripe banana?",
    "3. What is the chemical symbol for gold?",
    "4. How many legs does a spider have?",
    "5. What is the capital city of Japan?"
]

from llmtools import chat, ParallelLLMProcessor
from functools import partial

chat_gpt = partial(chat,model = 'deepseek/deepseek-chat-v3-0324')

# Initialize and run processor
processor = ParallelLLMProcessor(chat_fn=chat_gpt, num_workers=3, save_to_db=True)
results = processor.process_prompts(test_prompts)
print(results)