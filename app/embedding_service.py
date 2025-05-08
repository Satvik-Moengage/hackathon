import os
from openai import AzureOpenAI

class EmbeddingService:
    """
    Handles the generation of text embeddings using Azure OpenAI.
    """

    def __init__(self):
        # Load configuration from environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError(
                "Azure OpenAI embedding configuration missing. "
                "Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT in your environment."
            )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )

    def generate_embeddings(self, text: str):
        """
        Generate embeddings for a given text string using Azure OpenAI.
        Returns the embedding vector (list of floats).
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name,
            )
            if not hasattr(response, "data") or not response.data:
                raise ValueError("No data in embedding response from Azure OpenAI.")

            embedding = response.data[0].embedding
            if not embedding:
                raise ValueError("Generated embedding is empty.")

            return embedding

        except Exception as e:
            # Optionally, you can add logging here
            print(f"[EmbeddingService] Error generating embeddings: {e}")
            raise
