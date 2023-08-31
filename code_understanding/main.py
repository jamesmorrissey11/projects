from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake
from pydantic import BaseModel
import os 

def build_retriever(model_dir):
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 3},
    )
    return retriever


def build_qa_chain(retriever, qa_chain_prompt):
    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": qa_chain_prompt}
    )
    return qa_chain


def build_qa_prompt():
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer: Lets take it step by step"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    return QA_CHAIN_PROMPT


app = FastAPI()


class QuestionItem(BaseModel):
    question: str


@app.get("/")
async def code_qa_item(item: QuestionItem):
    qa_chain_prompt = build_qa_prompt()
    retriever = build_retriever(model_dir="/Users/jamesmorrissey//projects/code_understanding/models/2023-08-31_01-20-43")
    qa_chain = build_qa_chain(retriever=retriever, qa_chain_prompt=qa_chain_prompt)
    result = qa_chain({"query": item.question})
    answer = result["result"]
    return answer
