import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
import pypdf
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(self):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-multilingual-v3.0"
        )
        self.db = Chroma(
            collection_name="guias",
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path: str) -> None:
        """Procesa un único archivo PDF"""
        try:
            logger.info(f"Procesando archivo: {file_path}")
            
            # Leer PDF y extraer texto
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                raw_text = ""
                for page in pdf.pages:
                    raw_text += page.extract_text() + "\n"
            
            # Limpiar texto
            processed_text = raw_text
            
            # Log para debugging
            logger.debug(f"Primeros 500 caracteres del texto procesado:\n{processed_text[:500]}")
            
            # Crear chunks
            logger.info("Creando chunks...")
            chunks = self.text_splitter.create_documents(
                [processed_text],
                metadatas=[{"source": file_path}]
            )
            
            logger.info(f"Creados {len(chunks)} chunks")
            
            # Almacenar en ChromaDB
            logger.info("Guardando en ChromaDB...")
            self.db.add_documents(chunks)
            
            logger.info(f"Archivo procesado exitosamente: {file_path}")
            
        except Exception as e:
            logger.error(f"Error procesando {file_path}: {str(e)}")
            raise

if __name__ == "__main__":
    # Configuración
    PDF_PATH = "teoria_pintura.pdf"  # Ruta a tu PDF
    
    # Crear ingester
    ingester = DocumentIngester()
    
    # Procesar documento
    try:
        ingester.process_pdf(PDF_PATH)
        logger.info("Proceso de ingesta completado exitosamente")
    except Exception as e:
        logger.error(f"Error durante la ingesta: {str(e)}")