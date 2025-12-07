"""
Test the HuggingFace Inference Endpoint.

Usage:
    python scripts/test_endpoint.py
    python scripts/test_endpoint.py --prompt "Your custom prompt here"

Environment Variables:
    HF_ENDPOINT_URL: The inference endpoint URL
    HF_TOKEN: HuggingFace API token for authentication
"""

import argparse
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def query_endpoint(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Query the HuggingFace Inference Endpoint."""
    endpoint_url = os.getenv("HF_ENDPOINT_URL")
    hf_token = os.getenv("HF_TOKEN")

    if not endpoint_url:
        raise ValueError("HF_ENDPOINT_URL environment variable is required")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "return_full_text": False,
        },
    }

    print(f"Querying endpoint: {endpoint_url}")
    print(f"Prompt: {prompt}\n")

    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()

    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", str(result))
    elif isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace Inference Endpoint")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Generate a technical screening question for a senior backend engineer:",
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    try:
        response = query_endpoint(args.prompt, args.max_tokens, args.temperature)
        print("=" * 50)
        print("Response:")
        print("=" * 50)
        print(response)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
