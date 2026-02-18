"""Google AI Embeddings ModelClient integration (google-genai SDK)."""

import os
import logging
import backoff
from typing import Dict, Any, Optional, List, Sequence

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai is required. Install it with 'pip install google-genai'")

log = logging.getLogger(__name__)


class GoogleEmbedderClient(ModelClient):
    __doc__ = r"""A component wrapper for Google AI Embeddings API client.

    This client provides access to Google's embedding models through the Google AI API
    using the new google-genai SDK.

    Args:
        api_key (Optional[str]): Google AI API key. Defaults to None.
            If not provided, will use the GOOGLE_API_KEY environment variable.
        env_api_key_name (str): Environment variable name for the API key.
            Defaults to "GOOGLE_API_KEY".

    Example:
        ```python
        from api.google_embedder_client import GoogleEmbedderClient
        import adalflow as adal

        client = GoogleEmbedderClient()
        embedder = adal.Embedder(
            model_client=client,
            model_kwargs={
                "model": "gemini-embedding-001",
                "task_type": "SEMANTIC_SIMILARITY"
            }
        )
        ```

    References:
        - Google AI Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
        - Available models: gemini-embedding-001
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_api_key_name: str = "GOOGLE_API_KEY",
    ):
        """Initialize Google AI Embeddings client.

        Args:
            api_key: Google AI API key. If not provided, uses environment variable.
            env_api_key_name: Name of environment variable containing API key.
        """
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google AI client with API key."""
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        self._client = genai.Client(api_key=api_key)

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """Parse Google AI embedding response to EmbedderOutput format.

        Args:
            response: Google AI embedding response from google-genai SDK

        Returns:
            EmbedderOutput with parsed embeddings
        """
        try:
            from adalflow.core.types import Embedding

            embedding_data = []

            # New SDK: response has .embeddings attribute (list of ContentEmbedding)
            # Each ContentEmbedding has .values (list of floats)
            if hasattr(response, "embeddings") and response.embeddings:
                for i, emb in enumerate(response.embeddings):
                    if hasattr(emb, "values") and emb.values:
                        embedding_data.append(Embedding(embedding=list(emb.values), index=i))
            elif isinstance(response, dict):
                # Fallback for dict-like responses
                embeddings = response.get("embeddings", [])
                if isinstance(embeddings, list):
                    for i, emb in enumerate(embeddings):
                        if isinstance(emb, dict) and "values" in emb:
                            embedding_data.append(Embedding(embedding=emb["values"], index=i))
                        elif isinstance(emb, list):
                            embedding_data.append(Embedding(embedding=emb, index=i))
                # Legacy format fallback
                embedding_val = response.get("embedding")
                if not embedding_data and embedding_val and isinstance(embedding_val, list):
                    if isinstance(embedding_val[0], (int, float)):
                        embedding_data = [Embedding(embedding=embedding_val, index=0)]
                    elif isinstance(embedding_val[0], list):
                        embedding_data = [
                            Embedding(embedding=emb_list, index=i)
                            for i, emb_list in enumerate(embedding_val)
                            if isinstance(emb_list, list) and len(emb_list) > 0
                        ]

            if embedding_data:
                first_dim = len(embedding_data[0].embedding) if embedding_data[0].embedding is not None else 0
                log.info("Parsed %s embedding(s) (dim=%s)", len(embedding_data), first_dim)
            else:
                log.warning("Empty or unexpected embedding response: %s", type(response))

            return EmbedderOutput(
                data=embedding_data,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"Error parsing Google AI embedding response: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=response
            )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to Google AI API format.

        Args:
            input: Text input(s) to embed
            model_kwargs: Model parameters including model name and task_type
            model_type: Should be ModelType.EMBEDDER for this client

        Returns:
            Dict: API kwargs for Google AI embedding call
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"GoogleEmbedderClient only supports EMBEDDER model type, got {model_type}")

        if isinstance(input, str):
            contents = [input]
        elif isinstance(input, Sequence):
            contents = list(input)
        else:
            raise TypeError("input must be a string or sequence of strings")

        final_model_kwargs = model_kwargs.copy()
        final_model_kwargs["contents"] = contents

        # Set default task type if not provided
        if "task_type" not in final_model_kwargs:
            final_model_kwargs["task_type"] = "SEMANTIC_SIMILARITY"

        # Set default model if not provided
        if "model" not in final_model_kwargs:
            final_model_kwargs["model"] = "gemini-embedding-001"

        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call Google AI embedding API.

        Args:
            api_kwargs: API parameters
            model_type: Should be ModelType.EMBEDDER

        Returns:
            Google AI embedding response
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"GoogleEmbedderClient only supports EMBEDDER model type")

        # Work on a copy to avoid mutating the original dict (safe for backoff retries)
        kwargs = api_kwargs.copy()

        model = kwargs.pop("model", "gemini-embedding-001")
        contents = kwargs.pop("contents", None)
        task_type = kwargs.pop("task_type", None)
        output_dimensionality = kwargs.pop("output_dimensionality", None)

        if contents is None:
            raise ValueError("'contents' must be provided in api_kwargs")

        safe_log_kwargs = {"model": model, "task_type": task_type}
        if isinstance(contents, list):
            safe_log_kwargs["contents_count"] = len(contents)
        else:
            safe_log_kwargs["content_chars"] = len(str(contents))
        log.info("Google AI Embeddings call kwargs (sanitized): %s", safe_log_kwargs)

        try:
            # Build config with optional parameters
            config_kwargs = {}
            if task_type:
                config_kwargs["task_type"] = task_type
            if output_dimensionality:
                config_kwargs["output_dimensionality"] = output_dimensionality

            config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

            response = self._client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )

            return response

        except Exception as e:
            log.error(f"Error calling Google AI Embeddings API: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Async call to Google AI embedding API.

        Note: Falls back to synchronous call.
        """
        return self.call(api_kwargs, model_type)
