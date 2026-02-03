from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store #from utils.py
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

DATA_DIR = os.getenv("DATA_DIR", "data")

#step 1
def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".md":
                for d in UnstructuredMarkdownLoader(path).load(): #to load .md files, get data, and store as lang docs.
                    docs.append(d) #we simply add in to the docs which is a list.
            elif ext == ".pdf":
                for d in PyMuPDFLoader(path).load():
                    docs.append(d)
            elif ext == ".docx":
                for d in UnstructuredWordDocumentLoader(path).load(): 
                    docs.append(d)
            elif ext == ".txt":
                for d in TextLoader(path).load(): 
                    docs.append(d)
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs
        
#step 2
def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
       chunk_size = 900,
       chunk_overlap = 120
    )
    try:
        return splitter.split_documents(docs)# we receive list of docs & we split even more chunks then return the final List[] of docs. 
    except Exception:
        print(f"INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise


#step 3
async def run_ingest_async() -> dict:
   docs = _load_docs()
   chunks = _chunk(docs)
   store = await get_vector_store() #from utils.py for the db access ; already imported ; now got access for db.
   await store.aadd_documents(chunks) #taking the chunks; calc the embedings using the embeding model(from utils.py); store these chunks & embedings in vectore store table.
   print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks") #for the console. 

   return {"documents": len(docs), "chunks":len(chunks)} #for the ui: a dictory will contain docs and chunks.
