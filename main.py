import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def main() -> None:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a company chatbot",
            },
            {
                "role": "user",
                "content": "What does our company policy say?",
            },
        ],
    )

    print(
        completion.choices[0].message.content,
    )


if __name__ == "__main__":
    main()
