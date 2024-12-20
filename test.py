from langchain_chroma import Chroma
from app.core.config import settings
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY, model="embed-multilingual-v3.0"
)

vectorstore = Chroma(
    collection_name="productos",
    persist_directory=settings.CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
)


# Ver cantidad de documentos
print(vectorstore._collection.count())

# Ver muestra de documentos
print(vectorstore._collection.peek())

# Ver todos los documentos
results = vectorstore._collection.get()
print(results['documents'])  # Contenido
print(results['metadatas'])  # Metadata asociada
