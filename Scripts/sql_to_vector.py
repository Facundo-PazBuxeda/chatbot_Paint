from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from sqlalchemy import create_engine
import pandas as pd
from app.core.config import settings

engine = create_engine(settings.MYSQL_URL)
df = pd.read_sql("""
    SELECT p.marca, p.nombre, p.precio_regular, p.precio_promo, c.nombre as categoria
    FROM productos p
    LEFT JOIN categorias c ON p.categoria_id = c.id
    LIMIT 160
""", engine)

# Reemplazar valores None
df = df.fillna('')

documents = []
metadatas = []
ids = []

for i, row in df.iterrows():
    text = f"Marca: {row['marca']} | Producto: {row['nombre']} | Categoría: {row['categoria'] or 'Sin categoría'} | Precio: ${row['precio_regular'] or '0'} | Precio Promocional: ${row['precio_promo'] or '0'}"
    documents.append(text)
    ids.append(f"prod_{i}")
    metadatas.append({
        'marca': str(row['marca'] or ''),
        'nombre': str(row['nombre'] or ''),
        'categoria': str(row['categoria'] or 'Sin categoría'),
        'precio_regular': str(row['precio_regular'] or '0'),
        'precio_promo': str(row['precio_promo'] or '0')
    })

embeddings = CohereEmbeddings(
    cohere_api_key=settings.COHERE_API_KEY, 
    model="embed-multilingual-v3.0"
)
vectorstore = Chroma(
    collection_name="productos",
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIR
)

vectorstore.add_texts(
    texts=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Agregados {len(documents)} productos a la colección vectorial")
print("Collection stats:")
print(vectorstore._collection.count())
print(vectorstore._collection.peek())
