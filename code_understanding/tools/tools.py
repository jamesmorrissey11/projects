from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.vectorstores import DeepLake


def create_web_search_tool(serper_api_key) -> Tool:
    web_search = GoogleSerperAPIWrapper(
        serper_api_key=serper_api_key,
    )
    web_search_tool = Tool(
        name="Web Search",
        func=web_search.run,
        description="useful for when you need to answer questions about current information. input should be a fully formed question",
    )
    return web_search_tool


def create_vectorstore_search_tool(model_dir) -> Tool:
    retrieval_llm = OpenAI(temperature=0)
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    vectorstore_retriever = RetrievalQA.from_chain_type(
        llm=retrieval_llm, retriever=db.as_retriever(), chain_type="stuff"
    )
    vectorstore_search_tool = Tool(
        name="Search",
        func=vectorstore_retriever.run,
        description="useful for when you need to answer questions about langchain using code files directly from the repository. input should be a fully formed question",
    )
    return vectorstore_search_tool
