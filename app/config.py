from pydantic_settings import BaseSettings # Changed from pydantic to pydantic_settings
from pydantic import Field, HttpUrl


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")

    QDRANT_HOST: str = Field("172.16.0.25", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")

    EMBEDDING_MODEL_NAME: str = Field(..., env="EMBEDDING_MODEL_NAME")

    AZURE_OPENAI_API_KEY: str = Field(..., env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: HttpUrl = Field(..., env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(..., env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = Field("2023-05-15", env="AZURE_OPENAI_API_VERSION")

    ZENDESK_SUBDOMAIN: str = Field(..., env="ZENDESK_SUBDOMAIN")
    ZENDESK_EMAIL: str = Field(..., env="ZENDESK_EMAIL")
    ZENDESK_API_TOKEN: str = Field(..., env="ZENDESK_API_TOKEN")

    EXTERNAL_QDRANT_HOST: str = Field(..., env="EXTERNAL_QDRANT_HOST")
    EXTERNAL_QDRANT_PORT: int = Field(6333, env="EXTERNAL_QDRANT_PORT")
    EXTERNAL_CONFLUENCE_COLLECTION_NAME: str = Field(..., env="EXTERNAL_CONFLUENCE_COLLECTION_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()