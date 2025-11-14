# Backend

## main.py

::: main.root

## api/endpoints.py

::: api.endpoints.QueryRequest
::: api.endpoints.ask_text
::: api.endpoints.ask_mixed

## core/create_collection.py

::: core.create_collection.create_collection

## core/embeddings.py

::: core.embeddings.text_embedding
::: core.embeddings.image_embedding_from_path

## core/llm.py

::: core.llm.QwenVisionLLM
::: core.llm.QwenVisionLLM.__init__
::: core.llm.QwenVisionLLM.build_messages
::: core.llm.QwenVisionLLM.generate
::: core.llm.get_llm_response

## core/load_data.py

::: core.load_data.chunk_text
::: core.load_data.process_file
::: core.load_data.load_documents_to_qdrant

## core/multimodal_rag.py

::: core.multimodal_rag.LocalRAG
::: core.multimodal_rag.LocalRAG.__init__
::: core.multimodal_rag.LocalRAG.retrieve
::: core.multimodal_rag.LocalRAG.generate_answer

## core/vectordb.py

::: core.vectordb.get_qdrant
::: core.vectordb.ensure_collection_text
::: core.vectordb.ensure_collection_image
::: core.vectordb.add_text_points
::: core.vectordb.add_image_point
::: core.vectordb.search_text
::: core.vectordb.search_images
