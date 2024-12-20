from pydantic_settings import BaseSettings
import dotenv

dotenv.load_dotenv()

class Settings(BaseSettings):
    MYSQL_URL: str
    CHROMA_PERSIST_DIR: str
    COHERE_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()