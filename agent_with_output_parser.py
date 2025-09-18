from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain import hub
from schemas import AgentResponse, Source
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS


tools = [TavilySearch()]
llm = ChatOllama(temperature=0, model="gemma3")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def parse_output(output: str) -> AgentResponse:
    try:
        # Extract final answer from agent output
        if isinstance(output, dict) and "output" in output:
            output = output["output"]

        # Extract URLs from the text
        import re

        urls = re.findall(r"https?://[^\s\)]+", output)
        sources = [Source(url=url) for url in urls]

        return AgentResponse(answer=output, sources=sources)
    except Exception as e:
        print(f"Parsing error: {e}")
        return AgentResponse(answer=str(output), sources=[])


# extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
# chain = agent_executor | extract_output | parse_output
chain = agent_executor | RunnableLambda(parse_output)


def main():
    # print(react_prompt, type(react_prompt))
    # print(type(agent))
    result = chain.invoke(
        {
            "input": "Search for job postings for Python and AI engineer based in Kolkata on LinkedIn.",
        }
    )
    print(result)
    print(type(result))


if __name__ == "__main__":
    main()
