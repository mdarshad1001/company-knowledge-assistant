import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

PG_CONN_STR = os.getenv("DATABASE_URL")

#to create a vector_store object(establish connection with this)
PGENGINE = PGEngine.from_connection_string(PG_CONN_STR)

#embedings object
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


async def get_vector_store()->AsyncPGVectorStore: #get_vector_store is a method that will return vector store object
    #pass
    return await AsyncPGVectorStore.create(
        engine=PGENGINE,
        embedding_service=embeddings,
        table_name="langchain_pg_embedding"
    )

