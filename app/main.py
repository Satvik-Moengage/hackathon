import google.generativeai as genai

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint # Explicit import as in original

from typing import Optional, Dict, Any, List
import httpx
import asyncio

# Assuming app.embedding_service exists
from app.embedding_service import EmbeddingService
# class EmbeddingService: # Dummy for local testing if needed
#     def __init__(self): print("Dummy EmbeddingService initialized.")
#     def generate_embeddings(self, text: str) -> List[float]: return [0.1] * 10

load_dotenv(dotenv_path="../.env")

# Primary Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Zendesk
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN") # Expected to be pre-base64 encoded

# --- Configuration for the Second Qdrant DB (for Confluence) ---
EXTERNAL_QDRANT_HOST = os.getenv("EXTERNAL_QDRANT_HOST")
EXTERNAL_QDRANT_PORT = int(os.getenv("EXTERNAL_QDRANT_PORT", 6333))
EXTERNAL_CONFLUENCE_COLLECTION_NAME = os.getenv("EXTERNAL_CONFLUENCE_COLLECTION_NAME", "vectordbinternal")
# ---

app = FastAPI(
    title="Dynamic Model & Multi-Source KB Chatbot API",
    description="API for a Knowledge Base Chatbot with Zendesk ticket analysis, solution generation, and multi-Qdrant support. Returns a single consolidated answer.",
    version="0.9.1" # Version reflecting consolidated response
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant Clients
qdrant_client_primary: Optional[QdrantClient] = None
qdrant_client_external: Optional[QdrantClient] = None

# Other Services
embedding_service: Optional[EmbeddingService] = None
httpx_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global qdrant_client_primary, qdrant_client_external, embedding_service, httpx_client

    if not GOOGLE_API_KEY: raise RuntimeError("GOOGLE_API_KEY not configured.")
    if not QDRANT_HOST: print("Warning: Primary QDRANT_HOST not configured. Primary Qdrant DB will be unavailable.")
    if not ZENDESK_SUBDOMAIN or not ZENDESK_EMAIL or not ZENDESK_API_TOKEN:
        print("Warning: Zendesk configuration missing. Zendesk features will be limited.")

    if QDRANT_HOST:
        try:
            qdrant_client_primary = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
            qdrant_client_primary.get_collections()
            print(f"Successfully connected to Primary Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}.")
        except Exception as e:
            print(f"Error connecting to Primary Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}. Primary Qdrant functionalities will be unavailable.")
            qdrant_client_primary = None
    else:
        qdrant_client_primary = None

    if EXTERNAL_QDRANT_HOST and EXTERNAL_CONFLUENCE_COLLECTION_NAME:
        try:
            qdrant_client_external = QdrantClient(
                host=EXTERNAL_QDRANT_HOST,
                port=EXTERNAL_QDRANT_PORT,
                check_compatibility=False,
            )
            qdrant_client_external.get_collection(collection_name=EXTERNAL_CONFLUENCE_COLLECTION_NAME)
            print(f"Successfully connected to External Qdrant server (for Confluence) at {EXTERNAL_QDRANT_HOST}:{EXTERNAL_QDRANT_PORT}, collection '{EXTERNAL_CONFLUENCE_COLLECTION_NAME}' accessible.")
        except Exception as e:
            print(f"Error connecting to External Qdrant at {EXTERNAL_QDRANT_HOST}:{EXTERNAL_QDRANT_PORT} or accessing collection '{EXTERNAL_CONFLUENCE_COLLECTION_NAME}': {e}. Confluence search via external Qdrant may not be available.")
            qdrant_client_external = None
    else:
        print("External Qdrant (for Confluence) not configured. Confluence search will be unavailable.")
        qdrant_client_external = None

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Generative AI SDK configured successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to configure Google AI SDK: {e}")

    try:
        embedding_service = EmbeddingService()
        print("Azure OpenAI EmbeddingService initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize EmbeddingService: {e}")

    httpx_client = httpx.AsyncClient()
    print("HTTPX client initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    global qdrant_client_primary, qdrant_client_external, httpx_client
    if qdrant_client_primary:
        qdrant_client_primary.close()
        print("Primary Qdrant client closed.")
    if qdrant_client_external:
        qdrant_client_external.close()
        print("External Qdrant client (for Confluence) closed.")
    if httpx_client:
        await httpx_client.aclose()
        print("HTTPX client closed.")

class QueryRequest(BaseModel):
    question: str
    collection_names: List[str]
    generative_model_name: str
    top_k: int = 3

class ContextDetail(BaseModel):
    text: str
    source_collection: str
    score: float
    conversation: Optional[List[Dict[str, Any]]] = None
    ticket_id: Optional[str] = None
    ai_generated_solutions: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_context: List[ContextDetail]
    model_used: str

def get_embedding(text: str) -> List[float]:
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not initialized.")
    try:
        return embedding_service.generate_embeddings(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {e}")

async def fetch_zendesk_conversation(ticket_id: str) -> Optional[List[Dict[str, Any]]]:
    if not httpx_client or not ZENDESK_SUBDOMAIN or not ZENDESK_EMAIL or not ZENDESK_API_TOKEN:
        print("Zendesk client or configuration not available for fetching conversation.")
        return None
    headers = {"Authorization": f"Basic {ZENDESK_API_TOKEN}"}
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/comments.json"
    try:
        print(f"Fetching comments for Zendesk ticket: {ticket_id} from {url}")
        response = await httpx_client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return data.get("comments", [])
    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching Zendesk ticket {ticket_id} comments: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Error fetching Zendesk ticket {ticket_id} comments: {e}")
    return None

async def analyze_ticket_conversation_with_ai(
    ticket_id: str,
    conversation_comments: List[Dict[str, Any]],
    generative_model_id: str
) -> Optional[str]:
    if not conversation_comments: return None
    print(f"Analyzing conversation for ticket {ticket_id} with LLM '{generative_model_id}'...")
    formatted_conversation = []
    for comment in conversation_comments:
        author_id = comment.get('author_id')
        author_name = f"User {author_id}"
        if not comment.get('public', True): author_name = f"Agent (Internal Note)"
        elif 'user_id' in comment and comment['user_id'] == author_id : author_name = f"Customer (ID: {author_id})"
        else: author_name = f"Agent (ID: {author_id})"
        body = comment.get('plain_body', comment.get('body', '')).strip()
        if body: formatted_conversation.append(f"{author_name}: {body}")
    if not formatted_conversation:
        print(f"No textual content in conversation for ticket {ticket_id}.")
        return None
    conversation_str = "\n".join(formatted_conversation)
    prompt = f"""You are an expert technical support analyst.
Analyze the following Zendesk ticket conversation for ticket ID '{ticket_id}'.
Based SOLELY on this conversation, identify the core problem and provide a concise, actionable list of 2-3 potential solutions or troubleshooting steps.
If the conversation is too vague, say "The conversation does not provide enough information to suggest specific solutions."
Ticket Conversation:
---
{conversation_str}
---
Potential Solutions:"""
    try:
        model_to_use = genai.GenerativeModel(model_name=generative_model_id)
        response = await model_to_use.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing ticket {ticket_id} with LLM '{generative_model_id}': {e}")
        return None

async def search_primary_qdrant(query_embedding: List[float], collection_names: List[str], top_k: int, generative_model_name_for_analysis: str) -> List[ContextDetail]:
    if not qdrant_client_primary:
        print("Primary Qdrant client not available.")
        return []

    all_hits_with_collection_info: List[tuple[ScoredPoint, str]] = []
    for collection_name_str in collection_names:
        current_collection_name = collection_name_str.strip()
        if not current_collection_name: continue
        try:
            print(f"Searching Primary Qdrant collection: {current_collection_name}")
            search_results = qdrant_client_primary.search(
                collection_name=current_collection_name,
                query_vector=query_embedding,
                limit=top_k, with_payload=True
            )
            for hit in search_results: # hit is ScoredPoint
                all_hits_with_collection_info.append((hit, current_collection_name))
        except Exception as e:
            print(f"Warning: Error searching Primary Qdrant collection '{current_collection_name}': {e}")
            continue
    
    if not all_hits_with_collection_info: return []
    all_hits_with_collection_info.sort(key=lambda x: x[0].score, reverse=True)

    raw_context_details: List[ContextDetail] = []
    seen_texts_or_ids = set()
    
    for scored_point, source_collection_short_name in all_hits_with_collection_info:
        payload = scored_point.payload or {}
        text_content = payload.get("text")
        ticket_id_str = str(payload.get("ticket_id")) if payload.get("ticket_id") is not None else None
        
        unique_key = ticket_id_str if ticket_id_str else text_content # Key for deduplication
        if not unique_key or unique_key in seen_texts_or_ids:
            continue
            
        conversation_data: Optional[List[Dict[str, Any]]] = None
        full_source_collection_name = f"{source_collection_short_name}_primary_qdrant"
            
        if source_collection_short_name.lower() == "zendesk" and ticket_id_str and ZENDESK_SUBDOMAIN:
            conversation_data = await fetch_zendesk_conversation(ticket_id_str)

        raw_context_details.append(
            ContextDetail(
                text=text_content if text_content else "N/A",
                source_collection=full_source_collection_name,
                score=scored_point.score,
                conversation=conversation_data,
                ticket_id=ticket_id_str,
                title=payload.get("title"),
                url=payload.get("url") 
            )
        )
        if unique_key: seen_texts_or_ids.add(unique_key)

    analysis_tasks = []
    details_for_analysis_indices = []
    for i, detail in enumerate(raw_context_details):
        if "zendesk" in detail.source_collection.lower() and detail.conversation and detail.ticket_id:
            task = analyze_ticket_conversation_with_ai(
                detail.ticket_id,
                detail.conversation,
                generative_model_name_for_analysis
            )
            analysis_tasks.append(task)
            details_for_analysis_indices.append(i)

    if analysis_tasks:
        print(f"Starting AI analysis for {len(analysis_tasks)} Zendesk tickets from Primary Qdrant...")
        ai_solutions_results = await asyncio.gather(*analysis_tasks)
        result_idx = 0
        for i, detail_idx in enumerate(details_for_analysis_indices):
            raw_context_details[detail_idx].ai_generated_solutions = ai_solutions_results[i]
    else:
        print("No Zendesk tickets from Primary Qdrant with conversations found for AI analysis.")
    return raw_context_details

async def search_external_qdrant_for_confluence(query_embedding: List[float], top_k: int) -> List[ContextDetail]:
    if not qdrant_client_external or not EXTERNAL_CONFLUENCE_COLLECTION_NAME:
        print("External Qdrant client (for Confluence) not available or collection name not configured.")
        return []

    retrieved_details: List[ContextDetail] = []
    try:
        print(f"Searching External Qdrant (Confluence) collection: {EXTERNAL_CONFLUENCE_COLLECTION_NAME}")
        search_results = qdrant_client_external.search(
            collection_name=EXTERNAL_CONFLUENCE_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        seen_urls = set()
        for hit in search_results: # hit is ScoredPoint
            payload = hit.payload or {}
            text_content = payload.get("page_content")
            metadata = payload.get("metadata", {}) # Get metadata dictionary
            title = metadata.get("title")
            url = metadata.get("source") # 'source' field in metadata contains the URL

            if url and url in seen_urls: # Deduplicate by URL
                continue
            
            if text_content:
                retrieved_details.append(
                    ContextDetail(
                        text=text_content,
                        source_collection=f"confluence_external_qdrant",
                        score=hit.score,
                        title=title,
                        url=url
                    )
                )
                if url: seen_urls.add(url)
    except Exception as e:
        print(f"Error searching External Qdrant (Confluence) collection '{EXTERNAL_CONFLUENCE_COLLECTION_NAME}': {e}")
    return retrieved_details

def generate_answer_with_context(
    question: str,
    context_details: List[ContextDetail],
    generative_model_id: str
) -> str:
    if not context_details:
        prompt = f"Answer the following question. You have no specific context from the knowledge base.\nQuestion: {question}\nAnswer:"
    else:
        context_items = []
        for i, cd in enumerate(context_details):
            item_str = f"Context Snippet {i+1} from '{cd.source_collection}' (Score: {cd.score:.4f}):\nText: {cd.text}"
            if cd.title: item_str += f"\nTitle: {cd.title}"
            if cd.url: item_str += f"\nURL: {cd.url}"
            if cd.ticket_id: item_str += f"\n(Related Ticket ID: {cd.ticket_id})"
            if cd.conversation:
                convo_summary = "\nConversation Summary (first few messages):\n"
                for c_idx, comment in enumerate(cd.conversation[:2]):
                    author_name = comment.get('author_name', f"User {comment.get('author_id', 'Unknown')}")
                    comment_body = comment.get('plain_body', comment.get('body', 'No content')).strip()
                    convo_summary += f"  - {author_name}: {comment_body[:100]}...\n"
                item_str += convo_summary
            if cd.ai_generated_solutions:
                item_str += f"\nAI Suggested Solutions for Ticket {cd.ticket_id}:\n{cd.ai_generated_solutions}\n"
            context_items.append(item_str)
        context_str = "\n\n---\n\n".join(context_items)
        prompt = f"""You are a helpful AI assistant. Answer the user's question based *only* on the provided context.
The context may include text snippets from various sources (e.g., knowledge base articles, past tickets), summaries of ticket conversations, and AI-suggested general solutions derived from analyzing past tickets.
Synthesize all this information to provide a comprehensive answer.
If AI-suggested general solutions are provided for a ticket, consider how these general solutions might help answer the current user's question, especially if the problems seem related.
If the context doesn't contain the answer, clearly state that the information is not available from the provided documents.
Do not make up information. Do not include ticket IDs or any other identifiers in your answer. Also make your answer as general as possible, avoiding specific ticket references.

Provided Context:
---
{context_str}
---

Question: {question}

Answer:"""
    try:
        model_to_use = genai.GenerativeModel(model_name=generative_model_id)
        response = model_to_use.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer with LLM '{generative_model_id}': {e}")
        if "not found" in str(e).lower() or "permission" in str(e).lower() or "invalid" in str(e).lower():
             raise HTTPException(status_code=400, detail=f"Error with LLM '{generative_model_id}': Model not found or access denied. Original error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer with LLM '{generative_model_id}': {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_kb(request: QueryRequest):
    if not request.collection_names:
        raise HTTPException(status_code=400, detail="No collection names provided.")
    if not request.generative_model_name:
        raise HTTPException(status_code=400, detail="No generative model name provided.")

    print(f"Received question: '{request.question}' for collections: {request.collection_names} using model: {request.generative_model_name}")

    query_embedding = get_embedding(request.question)
    all_retrieved_context_details: List[ContextDetail] = []

    primary_qdrant_target_collections = [cn.strip() for cn in request.collection_names if cn.strip().lower() != "confluence"]
    should_search_confluence_externally = "confluence" in [cn.strip().lower() for cn in request.collection_names]

    if primary_qdrant_target_collections and qdrant_client_primary:
        print(f"Dispatching search to Primary Qdrant for collections: {primary_qdrant_target_collections}")
        primary_qdrant_results = await search_primary_qdrant(
            query_embedding,
            primary_qdrant_target_collections,
            request.top_k,
            request.generative_model_name
        )
        all_retrieved_context_details.extend(primary_qdrant_results)

    if should_search_confluence_externally and qdrant_client_external:
        print("Dispatching search to External Qdrant for 'confluence'")
        confluence_results = await search_external_qdrant_for_confluence(
            query_embedding,
            request.top_k
        )
        all_retrieved_context_details.extend(confluence_results)
    elif should_search_confluence_externally and not qdrant_client_external:
        print("Skipping Confluence search: External Qdrant client not available.")

    if not all_retrieved_context_details:
        print("No relevant context found in any vector database.")
        answer_no_context = generate_answer_with_context(request.question, [], request.generative_model_name)
        return ChatResponse(
            answer=answer_no_context,
            sources=[],
            retrieved_context=[],
            model_used=request.generative_model_name
        )
    
    all_retrieved_context_details.sort(key=lambda x: x.score, reverse=True)
    final_context_details = all_retrieved_context_details[:request.top_k]

    print(f"Total {len(final_context_details)} context snippets selected after merging and global top_k.")

    answer = generate_answer_with_context(
        request.question,
        final_context_details,
        request.generative_model_name
    )
    print(f"Generated final answer: {answer[:200]}...")

    compiled_sources: List[str] = []
    seen_source_links = set()
    for detail in final_context_details:
        source_link: Optional[str] = None
        if "zendesk" in detail.source_collection.lower() and detail.ticket_id:
            if ZENDESK_SUBDOMAIN:
                source_link = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{detail.ticket_id}"
            else: # Fallback if ZENDESK_SUBDOMAIN is not configured
                source_link = f"Zendesk Ticket ID: {detail.ticket_id}"
        elif "confluence" in detail.source_collection.lower() and detail.url:
            source_link = detail.url
        elif detail.url: # Generic fallback for other sources with a URL
            source_link = detail.url
        
        if source_link and source_link not in seen_source_links:
            compiled_sources.append(source_link)
            seen_source_links.add(source_link)

    return ChatResponse(
        answer=answer,
        sources=compiled_sources,
        retrieved_context=final_context_details,
        model_used=request.generative_model_name
    )

@app.get("/")
async def root():
    primary_q_status = "Not Configured"
    if qdrant_client_primary:
        try: qdrant_client_primary.get_collections(); primary_q_status = f"Connected ({QDRANT_HOST}:{QDRANT_PORT})"
        except: primary_q_status = f"Connection issue ({QDRANT_HOST}:{QDRANT_PORT})"
    elif QDRANT_HOST:
        primary_q_status = f"Configured ({QDRANT_HOST}) but client init failed or not available"
    
    external_q_status = "Not Configured"
    if qdrant_client_external and EXTERNAL_CONFLUENCE_COLLECTION_NAME:
        try: 
            qdrant_client_external.get_collection(EXTERNAL_CONFLUENCE_COLLECTION_NAME)
            external_q_status = f"Connected ({EXTERNAL_QDRANT_HOST}:{EXTERNAL_QDRANT_PORT}, Collection: {EXTERNAL_CONFLUENCE_COLLECTION_NAME})"
        except: 
            external_q_status = f"Connection issue or collection '{EXTERNAL_CONFLUENCE_COLLECTION_NAME}' not found ({EXTERNAL_QDRANT_HOST}:{EXTERNAL_QDRANT_PORT})"
    elif EXTERNAL_QDRANT_HOST:
        external_q_status = f"Configured ({EXTERNAL_QDRANT_HOST}) but client init failed or collection issue"

    g_sdk_status = "Configured" if GOOGLE_API_KEY else "Not Configured"
    es_status = "Initialized" if embedding_service else "Not Initialized"
    zd_config_ok = all([ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN])
    zd_status = "Configured & HTTPX Client Initialized" if zd_config_ok and httpx_client else \
                "Not Fully Configured" if not zd_config_ok else \
                "Configured but HTTPX client issue"

    return {
        "message": "Dynamic Model & Multi-Source KB Chatbot API (Dual Qdrant Support - Consolidated Response)",
        "primary_qdrant_status": primary_q_status,
        "external_qdrant_status (for Confluence)": external_q_status,
        "google_ai_sdk_status": g_sdk_status,
        "embedding_service_status": es_status,
        "zendesk_status": zd_status,
        "features": [
            "Multi-collection search in Primary Qdrant DB",
            "Conditional search in a Secondary Qdrant DB for 'confluence' data",
            "Azure OpenAI for embeddings",
            "Google Gemini for text generation",
            "Zendesk ticket conversation retrieval and AI-powered analysis",
            "Returns a single consolidated answer and a combined list of sources."
        ],
        "example_generative_models_to_try": [
            "gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"
        ],
        "expected_payload_for_chat": {
            "question": "How to resolve login issues?",
            "collection_names": ["zendesk", "confluence", "internal_docs"], 
            "generative_model_name": "gemini-1.5-flash-latest",
            "top_k": 5
        },
        "example_response_structure_for_chat": {
            "answer": "Based on the provided documents, login issues can often be resolved by resetting your password or checking network connectivity. Some confluence articles also mention clearing browser cache...",
            "sources": [
                "https://your-subdomain.zendesk.com/agent/tickets/12345",
                "https://your-confluence-space/page/login-troubleshooting-guide",
                "https://your-internal-docs/sso-faq" # Example if 'internal_docs' had a URL
            ],
            "retrieved_context": [
                {"text": "...", "source_collection": "zendesk_primary_qdrant", "score": 0.89, "ticket_id": "12345"},
                {"text": "...", "source_collection": "confluence_external_qdrant", "score": 0.85, "url": "..."}
            ],
            "model_used": "gemini-1.5-flash-latest"
        }
    }