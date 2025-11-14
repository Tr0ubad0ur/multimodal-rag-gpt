from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_text_collection: str = "qdrant_text"
    qdrant_image_collection: str = "qdrant_image"
    # Размеры векторов (важно для создания коллекций)
    text_vector_size: int = 1536
    image_vector_size: int = 512

    class Config:
        env_file = ".env"

settings = Settings()
