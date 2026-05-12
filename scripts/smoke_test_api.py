import os
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not found in .env")

print(f"Using model: {model}")
print("Calling Anthropic API...")

client = anthropic.Anthropic(api_key=api_key)
response = client.messages.create(
    model=model,
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say 'V3 RAG setup confirmed' in 5 different ways. Be brief."}
    ],
)

print("\n--- Response ---")
print(response.content[0].text)
print("\n--- Usage ---")
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Estimated cost: ${(response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000:.6f}")
