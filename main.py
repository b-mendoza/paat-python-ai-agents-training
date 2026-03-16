from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


class AgentMessage(BaseModel):
    content: str


def get_llm_message(llm: ChatOpenAI) -> AgentMessage:
    structured_llm_message = llm.with_structured_output(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] `with_structured_output()` returns `Runnable[..., _DictOrPydantic]`, but `_DictOrPydantic` uses a TypeVar (`_BM`) that Pyright can't resolve here.
        schema=AgentMessage,
        method="json_schema",
    )

    return structured_llm_message.invoke(
        input="What does our company policy say?",
    )  # pyright: ignore[reportReturnType, reportUnknownVariableType] `.invoke()` returns `AgentMessage` at runtime, but Pyright sees it as `dict | BaseModel` (the `_DictOrPydantic` alias), so it won't match our `-> AgentMessage` return type.


def main() -> None:
    class EnvVars(BaseModel):
        OPENAI_API_KEY: SecretStr

    validated_env_vars = EnvVars.model_validate(
        dotenv_values(dotenv_path=".env"),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        api_key=validated_env_vars.OPENAI_API_KEY,
    )

    llm_message = get_llm_message(llm)

    print(
        llm_message.content,
    )


if __name__ == "__main__":
    main()
