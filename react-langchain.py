from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import tool
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description


@tool
def length_tool(text: str) -> int:
    """Returns the length of the input text."""
    text = text.strip("'\n").strip('"')
    return len(text)


def main():
    # print(length_tool.invoke(input={"text": "Hello, world!"}))
    tools = [length_tool]
    prompt = PromptTemplate.from_template(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
    ).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )  # partial means we are fixing some variables in the prompt template, render_text_description(tools) generates a text description of the tools available for the agent to use
    llm = ChatOllama(
        temperature=0, model="gemma3", stop=["\nObservation"]
    )  # stop=["\nObservation"] means the LLM will stop generating text when it encounters the string "\nObservation"


if __name__ == "__main__":
    main()
