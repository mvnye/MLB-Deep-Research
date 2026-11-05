"""Ask the README's sample questions with no tools and no system prompt, and save the answers.

Each question goes to the model as a single user message. Answers are saved to samples/ for
comparison with the agent's answers.

Usage: uv run python scripts/baselines.py
"""

from datetime import date
from pathlib import Path

import litellm

MODEL = "vertex_ai/gemini-3.5-flash"
OUT = Path(__file__).resolve().parent.parent / "samples"
QUESTIONS = [
    "Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?",
    "Why did the Mets over-perform on attendance in 2025?",
    "How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?",
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for number, question in enumerate(QUESTIONS, 1):
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            vertex_project="agenticai-509716",
            vertex_location="global",
            timeout=90,
            num_retries=2,
        )
        answer = response.choices[0].message.content
        path = OUT / f"baseline_{number}.md"
        path.write_text(f"Model: {MODEL}\nDate: {date.today()}\nQuestion: {question}\n\n{answer}\n")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
