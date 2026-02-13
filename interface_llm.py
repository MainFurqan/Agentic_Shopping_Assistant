import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You extract shopping constraints.
Return JSON ONLY.

Fields:
budget (number)
delivery_deadline_days (number)
size (string or null)
style (string or null)
"""

def parse_user_message(message: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    user_message = "I need a new laptop for work. My budget is $1500, and I need it within 2 weeks. I prefer something lightweight and with a sleek design."
    constraints = parse_user_message(user_message)
    print(constraints)