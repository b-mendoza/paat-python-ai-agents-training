from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


class AgentMessage(BaseModel):
    content: str


def get_llm_message(llm: ChatOpenAI) -> AgentMessage:
    llm_message = llm.invoke("What does our company policy say?")

    return AgentMessage.model_validate(llm_message)


def main() -> None:
    class EnvVars(BaseModel):
        OPENAI_API_KEY: SecretStr

    validated_env_vars = EnvVars.model_validate(
        dotenv_values(
            dotenv_path=".env",
        ),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        api_key=validated_env_vars.OPENAI_API_KEY,
    )

    llm_message = get_llm_message(
        llm=llm,
    )

    print(
        llm_message.content,
    )


if __name__ == "__main__":
    main()
