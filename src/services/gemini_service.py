import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_MODEL = "gemini-3.5-flash"


def generate_financial_explanation(
    financial_summary: dict[str, Any],
) -> str:
    """
    Generate a plain-English explanation of verified financial analytics.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key is not configured.")

    client = genai.Client(api_key=api_key)

    summary_json = json.dumps(
        financial_summary,
        indent=2,
        default=str,
    )

    prompt = f"""
    You are a personal finance analytics assistant.

    Explain the following financial data in simple,
    understandable language.

    Instructions:
    1. Use only the financial figures provided.
    2. Never invent transactions, amounts, or statistics.
    3. Highlight important spending patterns.
    4. Explain income, expenses, and net balance.
    5. Offer practical, general budgeting suggestions.
    6. Do not present yourself as a financial advisor.
    7. If information is insufficient, clearly say so.
    8. Keep the explanation concise and structured.

    Verified financial data:

    {summary_json}
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()