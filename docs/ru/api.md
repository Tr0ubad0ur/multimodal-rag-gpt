# Backend API

## main.py

::: backend.main.root
::: backend.main.metrics

## api/endpoints.py

::: backend.api.endpoints.QueryRequest
::: backend.api.endpoints.TextEmbeddingRequest
::: backend.api.endpoints.ImageEmbeddingRequest
::: backend.api.endpoints.VideoEmbeddingRequest
::: backend.api.endpoints.ask_mixed
::: backend.api.endpoints.ask_mixed_auth
::: backend.api.endpoints.embed_text
::: backend.api.endpoints.embed_image
::: backend.api.endpoints.embed_video
::: backend.api.endpoints.upload_file
::: backend.api.endpoints.kb_tree
::: backend.api.endpoints.get_history

## core/embeddings.py

::: backend.core.embeddings.text_embedding
::: backend.core.embeddings.image_embedding_from_path
::: backend.core.embeddings.video_embedding_from_path

## core/llm.py

::: backend.core.llm.QwenVisionLLM
::: backend.core.llm.get_llm_response

## core/multimodal_rag.py

::: backend.core.multimodal_rag.LocalRAG
::: backend.core.multimodal_rag.LocalRAG.retrieve_data
::: backend.core.multimodal_rag.LocalRAG.generate_answer

## core/embedding_providers.py

::: backend.core.embedding_providers.EmbeddingProvider
::: backend.core.embedding_providers.SentenceTransformerProvider
::: backend.core.embedding_providers.get_provider

## utils/load_data.py

::: backend.utils.load_data.DataLoader

## monitoring/metrics.py

::: backend.monitoring.metrics.observe_http_request
::: backend.monitoring.metrics.observe_rag_query
::: backend.monitoring.metrics.observe_embedding_request
