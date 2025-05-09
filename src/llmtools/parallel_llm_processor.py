from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any
from dataclasses import dataclass
from tqdm import tqdm
import time

@dataclass
class OrderedResult:
    index: int
    result: str

class ParallelLLMProcessor:
    def __init__(self, chat_fn: Callable[[str], str], num_workers: int = 4, 
                 retry_attempts: int = 3, retry_delay: float = 1.0,
                 timeout: float = 60.0):
        """
        Initialize the parallel processor.
        
        Args:
            chat_fn: The LLM chat function that takes a prompt and returns a string
            num_workers: Number of parallel workers
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout in seconds for each request
        """
        self.chat_fn = chat_fn
        self.num_workers = num_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _process_single_prompt(self, args: tuple[int, str]) -> OrderedResult:
        """
        Process a single prompt with retry and timeout logic.
        
        Args:
            args: Tuple of (index, prompt)
            
        Returns:
            OrderedResult containing the original index and result
        """
        idx, prompt = args
        attempt = 0
        
        while attempt < self.retry_attempts:
            try:
                future = ThreadPoolExecutor(max_workers=1).submit(self.chat_fn, prompt)
                result = future.result(timeout=self.timeout)
                return OrderedResult(idx, result)
            except Exception as e:
                attempt += 1
                if attempt == self.retry_attempts:
                    raise TimeoutError(f"Failed after {self.retry_attempts} attempts: {str(e)}")
                time.sleep(self.retry_delay)

    def process_prompts(self, prompts: List[str]) -> List[str]:
        """
        Process multiple prompts in parallel while maintaining order.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            List of results in the same order as input prompts
        """
        results = [None] * len(prompts)
        tasks = list(enumerate(prompts))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_single_prompt, task): task[0]
                for task in tasks
            }
            
            with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
                for future in as_completed(futures):
                    try:
                        ordered_result = future.result()
                        results[ordered_result.index] = ordered_result.result
                    except Exception as e:
                        idx = futures[future]
                        results[idx] = f"Error: {str(e)}"
                    pbar.update(1)
        
        return results

# Example usage:
if __name__ == "__main__":
    import time
    
    def mock_chat_fn(prompt: str) -> str:
        """Mock LLM function for testing"""
        time.sleep(0.5)  # Simulate API delay
        return f"Response to: {prompt}"

    # Test prompts
    test_prompts = [
        "What is 8 squared?",
        "What color is a ripe banana?",
        "What is the chemical symbol for gold?",
        "How many legs does a spider have?",
        "What is the capital city of Japan?"
    ]
    
    # Initialize and run processor
    processor = ParallelLLMProcessor(chat_fn=mock_chat_fn, num_workers=3)
    results = processor.process_prompts(test_prompts)