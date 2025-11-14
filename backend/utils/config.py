from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        qdrant_url (str): URL of the Qdrant vector database.
        qdrant_text_collection (str): Name of the Qdrant collection for text embeddings. Defaults to "qdrant_text".
        qdrant_image_collection (str): Name of the Qdrant collection for image embeddings. Defaults to "qdrant_image".
        text_vector_size (int): Dimensionality of text embeddings. Defaults to 1536.
        image_vector_size (int): Dimensionality of image embeddings. Defaults to 512.
    """
    qdrant_url: str
    qdrant_text_collection: str = "qdrant_text"
    qdrant_image_collection: str = "qdrant_image"
    # Размеры векторов (важно для создания коллекций)
    text_vector_size: int = 1536
    image_vector_size: int = 512

    class Config:
        """Pydantic configuration class."""
        env_file = ".env"

settings = Settings()
