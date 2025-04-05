from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_mcqs(document_text: str, num_questions: int = 10):
    prompt_template = PromptTemplate.from_template("""
    Create {n} multiple-choice questions (MCQs) with 4 options and one correct answer based on the document below. 
    Format:
    Q: ...
    A. ...
    B. ...
    C. ...
    D. ...
    Answer: ...

    Document:
    {text}
    """)

    llm = ChatOpenAI(temperature=0.9)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(text=document_text, n=num_questions)
