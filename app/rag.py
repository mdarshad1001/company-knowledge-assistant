from typing import List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.docstore.document import Document

from langchain_core .globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from .utils import get_vector_store

from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever




SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Question:\n{input}\n\n"
     "Context:\n{context}\n\n"
     "Rule: Prefer the most recent policy by effective date.")
])

async def _build_chain(): #(1)
    store = await get_vector_store() #get vectore store.
    retriver = store.as_retriever(search_kwargs={"k":int(os.getenv("RETRIEVAL_K","5"))})
    llm = ChatOpenAI(model="gpt-4o-mini")
    #chains
    doc_chain = create_stuff_documents_chain(llm,PROMPT) #responsible for taking the chunks, stuffing it in the promt & handin it to LLM.
    rag_chain = create_retrieval_chain(retriver,doc_chain)

    return rag_chain

#this method will take question of type str and returns back a tuple str(answer) & list of str(source docs).
async def answer_with_docs_async(question: str) -> Tuple[str, List[str]]:
    chain = await _build_chain()
    result = await chain.ainvoke({"input": question}) #async invocation with the build chain. the result should have answer and context(list of lang docs) 
    #answer
    answer = result["answer"]
    
    #sources
    sources = []
    docs:list[Document]= result["context"] #each lang docs = will have chank & meta data
    unique_sources = {d.metadata.get("source") for d in docs} #looping throgh list of docs, retriving the source name , using {set} to remove duplicates.
    sources = sorted(unique_sources) #final source :sorted list

    return answer, sources
