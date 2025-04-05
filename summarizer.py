from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_summary(document_text: str):
    prompt_template = PromptTemplate.from_template("""
    Summarize the following text in a clear and concise way for students:

    {text}

    Summary:
    """)
    llm = ChatOpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(text=document_text)
