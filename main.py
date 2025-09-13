import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

load_dotenv()


def main():
    information = """Karenjit "Karen"[3][4] Kaur Vohra (born May 13, 1981), known by her stage name Sunny Leone, is a Canadian American actress, model, and former pornographic actress.[5][6] She was named Penthouse Pet of the Year in 2003, was a contract performer for Vivid Entertainment, and was named by Maxim as one of the 12 top porn stars in 2010. She was inducted into the AVN Hall of Fame in 2018.[7]

She has played roles in independent mainstream events, films, and television series. Her first mainstream appearance was in 2005, when she worked as a red carpet reporter for the MTV Video Music Awards on MTV India. In 2011, she participated in the Indian reality television series Bigg Boss. She also has hosted the Indian reality show Splitsvilla.

In 2012, she made her Bollywood debut in Pooja Bhatt's erotic thriller Jism 2 (2012) and shifted her focus to mainstream acting which was followed up with Jackpot (2013), Ragini MMS 2 (2014), Ek Paheli Leela (2015), Tera Intezaar (2017), and the Malayalam film Madhura Raja in 2019."""

    prompt = """Given the information {information} provided, do the following:
    1. Summarize the information
    2. Give 2 interesting facts"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=prompt
    )

    llm = ChatOllama(temperature=0, model="gemma3")

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
