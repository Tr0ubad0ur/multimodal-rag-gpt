from backend.core.llm import get_llm
from backend.core.vectordb import create_or_get_collection
from langchain_community.chains import RetrievalQA

def get_rag_chain():
    llm = get_llm()
    vector_store = create_or_get_collection()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return chain

def ask_rag(question: str):
    chain = get_rag_chain()
    answer = chain.run(question)
    return answer
