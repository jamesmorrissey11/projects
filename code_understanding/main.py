from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake
from pydantic import BaseModel
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os 

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


@app.post("/")
def code_qa_item(item: QuestionItem):
    model_dir ="/code/app/model"
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    relevant_docs = db.max_marginal_relevance_search(item.question)
    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answer = qa_chain.run(input_documents=relevant_docs, question=item.question)
    return {"answer": answer}
