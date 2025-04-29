def innit_app():    
    from llama_index.core.schema import TextNode
    from datasets import load_dataset
    import uuid

    # LOAD ENV
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # LLM
    from llama_index.llms.google_genai import GoogleGenAI

    # EMBEDDING
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # QDRANT (STORAGE)
    from llama_index.core.indices.vector_store.base import VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    import qdrant_client

    from llama_index.core.prompts import RichPromptTemplate
    from llama_index.core import PromptTemplate
    from llama_index.core.llms import ChatMessage, MessageRole
    
    from llama_index.core.postprocessor import SimilarityPostprocessor


    llm = GoogleGenAI(
        model="gemini-2.0-flash",
    )

    embed_model = HuggingFaceEmbedding(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )


    loaded_index = VectorStoreIndex.from_vector_store(
        QdrantVectorStore(
            client=qdrant_client.QdrantClient(
                "https://958364eb-7484-4a89-afae-40702093201e.us-east4-0.gcp.cloud.qdrant.io",
                api_key=os.environ["QDRANT_API_KEY"],
            ),
            collection_name="law_documents",
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20,
        ),
        embed_model=embed_model,
    )

    qa_prompt_tmpl_str = (
        "Bạn là trợ lý tư vấn pháp luật cho nhiệm vụ hỏi đáp với người dùng.\n"
        "Sử dụng các phần sau của bối cảnh được truy xuất để trả lời câu hỏi.\n"
        "Nếu bạn không biết câu trả lời, đừng cố tạo câu trả lời..\n"
        "Ngữ cảnh cung cấp:\n"
        "---------------------\n"
        "{{ context_str }}\n"
        "Hãy trả lời câu hỏi sau với phong cách của một luật sư.\n"
        "Người dùng hỏi: {{ query_str }}\n"
        "Trả lời: "
    )
    qa_prompt_tmpl = RichPromptTemplate(qa_prompt_tmpl_str)

    sim_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.4)

    custom_prompt ="""\
    Cho đoạn hội thoại(Giữa người dùng và trợ lý) và một câu hỏi tiếp theo từ người dùng, \
    vui lòng viết lại câu hỏi để nó trở thành một câu hỏi độc lập, \
    có thể hiểu được toàn bộ ngữ cảnh đoạn hội thoại. \

    <Đoạn hội thoại>
    {chat_history}

    <Câu hỏi tiếp theo>
    {question}

    <Câu hỏi độc lập>
    """
    

    custom_chat_history = [
        ChatMessage(
            role=MessageRole.USER,
            content="Chào bạn, tôi cần sự giúp đỡ của bạn về một vấn đề pháp lý, lĩnh vực hôn nhân gia đình.",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT, content="Được thôi! Tôi có thể giúp gì cho bạn?"
        ),
    ]

    query_engine = loaded_index.as_query_engine(
        text_qa_template=qa_prompt_tmpl,
        llm=llm,
        similarity_top_k=2,
        sparse_top_k=10,
        vector_store_query_mode="hybrid",
        node_postprocessors=[sim_postprocessor],
    )

    return {
        "query_engine": query_engine,
        "custom_prompt": custom_prompt,
        "custom_chat_history": custom_chat_history,
        "llm": llm,
    }


def create_chat_engine(query_engine,custom_prompt,custom_chat_history,llm):
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
    from llama_index.core import PromptTemplate
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt= PromptTemplate(custom_prompt),
        chat_history=custom_chat_history,
        verbose=True,
        llm=llm,
    )
    return chat_engine
