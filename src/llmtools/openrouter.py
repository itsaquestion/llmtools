from typing import Optional, Callable, List
import os
import json
import requests
import sys
from datetime import datetime
import dotenv

dotenv.load_dotenv()


def print_to_screen(token: str):
    """Default callback that prints tokens to screen."""
    print(token, end="", flush=True)


def save_to_file(filename: str) -> Callable:
    """
    Creates a callback that saves tokens to a file.

    Args:
        filename: Name of the file to save to
    Returns:
        Callable that saves tokens to the specified file
    """

    # Remove existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    def callback(token: str):
        with open(filename, "a", encoding="utf-8") as f:
            f.write(token)

    return callback


def dummy_callback(token: str):
    pass


def chat(
    prompt: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    callbacks: List[Callable] = [],
    provider: Optional[dict] = None,
    proxy: Optional[str] = None,
) -> str:
    """
    Send a streaming chat message to OpenRouter API with callbacks for token handling.

    Args:
        prompt: The text message to send
        model: The model identifier to use
        temperature: Controls randomness (0-2, default 1.0)
        max_tokens: Maximum tokens in response (optional)
        callbacks: List of callback functions to handle streaming tokens
        provider: Dictionary containing provider specific settings (optional)
        proxy: Proxy URL to use for the request. If not provided, will check environment variable LLMTOOLS_PROXY (optional)

    Returns:
        str: The complete response text
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or "http://localhost"
    x_title = os.getenv("OPENROUTER_X_TITLE") or "test app"

    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable")

    # Use default print callback if none provided
    if callbacks == []:
        callbacks = [dummy_callback]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer,
        "X-Title": x_title,
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": temperature,
        "stream": True,  # Enable streaming,
    }

    if provider:
        payload.update({"provider": provider})

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Handle proxy configuration -优先使用参数，然后是环境变量
    proxies = None
    if proxy is not None:
        proxies = {"http": proxy, "https": proxy}
    elif os.getenv("LLMTOOLS_PROXY"):
        proxies = {"http": os.environ["LLMTOOLS_PROXY"], "https": os.environ["LLMTOOLS_PROXY"]}

    try:
        full_response = ""
        with requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
            stream=True,
            proxies=proxies,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        if not data.get("choices"):
                            continue

                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                            full_response += token

                            # Process token through all callbacks
                            for callback in callbacks:
                                callback(token)

                    except json.JSONDecodeError:
                        continue

        return full_response

    except Exception as e:
        raise Exception(f"Chat failed: {str(e)}")


# Example usage:
if __name__ == "__main__":
    try:
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_{timestamp}.txt"

        # Setup callbacks
        screen_callback = print_to_screen
        file_callback = save_to_file(filename)

        # Run chat with both callbacks
        response = chat(
            "Say 'OK' and say 'OK' only./no_think",
            model="openai/gpt-4.1-nano",
            temperature=1,
            max_tokens=512,
            callbacks=[screen_callback],
            provider={"order": ["OpenAI", "Together"]}
        )

        # print(f"\n\nFull response saved to {filename}")

    except Exception as e:
        print(f"Error: {e}")
