import os
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
import google.generativeai as genai
from typing import Optional, Dict, Any, List
import httpx
import asyncio # For concurrent AI analysis

# Assuming app.embedding_service exists
from app.embedding_service import EmbeddingService
# class EmbeddingService: # Dummy for local testing if needed
#     def __init__(self): print("Dummy EmbeddingService initialized.")
#     def generate_embeddings(self, text: str) -> List[float]: return [0.1] * 10


load_dotenv(dotenv_path="../.env")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")

app = FastAPI(
    title="Dynamic Model & Multi-Collection KB Chatbot API",
    description="API for a Knowledge Base Chatbot with Zendesk ticket analysis and solution generation.",
    version="0.6.0" # Incremented version
)

qdrant_client: Optional[QdrantClient] = None
embedding_service: Optional[EmbeddingService] = None
httpx_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup_event():
    global qdrant_client, embedding_service, httpx_client
    # ... (startup checks for QDRANT, GOOGLE_API_KEY, ZENDESK credentials - keep as before)
    if not GOOGLE_API_KEY: raise RuntimeError("GOOGLE_API_KEY not configured.")
    if not QDRANT_HOST: raise RuntimeError("QDRANT_HOST configuration missing.")
    if not ZENDESK_SUBDOMAIN or not ZENDESK_EMAIL or not ZENDESK_API_TOKEN:
        raise RuntimeError("Zendesk configuration missing.")

    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
        qdrant_client.get_collections()
        print(f"Successfully connected to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}.")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        qdrant_client = None

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
    global qdrant_client, httpx_client
    if qdrant_client:
        qdrant_client.close()
        print("Qdrant client closed.")
    if httpx_client:
        await httpx_client.aclose()
        print("HTTPX client closed.")

class QueryRequest(BaseModel):
    question: str
    collection_names: List[str]
    generative_model_name: str # Used for both main answer and ticket analysis
    top_k: int = 3

class ContextDetail(BaseModel):
    text: str
    source_collection: str
    score: float
    conversation: Optional[List[Dict[str, Any]]] = None
    ticket_id: Optional[str] = None
    ai_generated_solutions: Optional[str] = None # NEW: For AI-analyzed solutions

class ChatResponse(BaseModel):
    answer: str
    retrieved_context: List[ContextDetail] # Will now include ai_generated_solutions
    model_used: str # For the main answer

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
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/comments.json"
    print(url)
    headers = {"Authorization": f"Basic {ZENDESK_API_TOKEN}"}
    try:
        print(f"Fetching comments for Zendesk ticket: {ticket_id}")
        response = await httpx_client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return data.get("comments", [])
    except Exception as e:
        print(f"Error fetching Zendesk ticket {ticket_id} comments: {e}")
    return None

async def analyze_ticket_conversation_with_ai(
    ticket_id: str,
    conversation_comments: List[Dict[str, Any]],
    generative_model_id: str
) -> Optional[str]:
    """
    Analyzes a ticket conversation using an LLM to suggest potential solutions.
    """
    if not conversation_comments:
        return None

    print(f"Analyzing conversation for ticket {ticket_id} with LLM '{generative_model_id}'...")

    # Prepare conversation for the prompt
    formatted_conversation = []
    for comment in conversation_comments:
        author_id = comment.get('author_id') # You might want to map this to role (e.g., agent/end-user) if possible
        author_name = f"User {author_id}" # Placeholder, Zendesk API might give more
        # Check if 'via' channel is 'API' or specific agent identifiers to mark as 'Agent'
        # This logic might need refinement based on your Zendesk comment structure
        if comment.get('public', True) == False: # Internal note
             author_name = f"Agent (Internal Note)"
        elif 'user_id' in comment and comment['user_id'] == author_id : # Crude check if author is the requester
             author_name = f"Customer (ID: {author_id})"
        else: # Assume agent if not clearly customer or internal
            author_name = f"Agent (ID: {author_id})"


        body = comment.get('plain_body', comment.get('body', '')).strip()
        if body: # Only include comments with actual text
            formatted_conversation.append(f"{author_name}: {body}")

    if not formatted_conversation:
        print(f"No textual content found in conversation for ticket {ticket_id} to analyze.")
        return None

    conversation_str = "\n".join(formatted_conversation)

    prompt = f"""You are an expert technical support analyst.
Analyze the following Zendesk ticket conversation for ticket ID '{ticket_id}'.
The user is likely facing an issue described in this conversation.
Based SOLELY on this conversation, identify the core problem and provide a concise, actionable list of 2-3 potential solutions or troubleshooting steps that could resolve the user's issue.
If the conversation is too vague or doesn't clearly state a problem, say "The conversation does not provide enough information to suggest specific solutions."
Do not invent information. Focus only on what is discussed in the ticket.

Ticket Conversation:
---
{conversation_str}
---

Potential Solutions:
"""
    try:
        model_to_use = genai.GenerativeModel(model_name=generative_model_id)
        # print(f"\n--- PROMPT FOR TICKET ANALYSIS (Ticket: {ticket_id}, Model: {generative_model_id}) ---")
        # print(prompt)
        # print("--- END OF TICKET ANALYSIS PROMPT ---\n")
        response = await model_to_use.generate_content_async(prompt) # Use async version
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing ticket {ticket_id} with LLM '{generative_model_id}': {e}")
        # Don't raise HTTPException here, just return None so main flow continues
        return None


async def search_qdrant(query_embedding: List[float], collection_names: List[str], top_k: int, generative_model_name_for_analysis: str) -> List[ContextDetail]:
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not available.")

    # ... (code for searching Qdrant and sorting hits - keep as before) ...
    all_hits_with_collection_info: List[tuple[ScoredPoint, str]] = []
    for collection_name_str in collection_names:
        current_collection_name = collection_name_str.strip()
        if not current_collection_name: continue
        try:
            search_results = qdrant_client.search(
                collection_name=current_collection_name,
                query_vector=query_embedding,
                limit=top_k, with_payload=True
            )
            for hit in search_results:
                all_hits_with_collection_info.append((hit, current_collection_name))
        except Exception as e:
            print(f"Warning: Error searching collection '{current_collection_name}': {e}")
            continue
    if not all_hits_with_collection_info: return []
    all_hits_with_collection_info.sort(key=lambda x: x[0].score, reverse=True)

    # Build initial ContextDetail list (without AI solutions yet)
    raw_context_details: List[ContextDetail] = []
    seen_texts = set()
    for scored_point, source_collection in all_hits_with_collection_info:
        if len(raw_context_details) >= top_k: break
        payload = scored_point.payload or {}
        text_content = payload.get("text")
        if text_content and text_content not in seen_texts:
            ticket_id_str = str(payload.get("ticket_id")) if payload.get("ticket_id") is not None else None
            conversation_data: Optional[List[Dict[str, Any]]] = None
            if source_collection.lower() == "zendesk" and ticket_id_str:
                conversation_data = await fetch_zendesk_conversation(ticket_id_str)

            raw_context_details.append(
                ContextDetail(
                    text=text_content,
                    source_collection=source_collection,
                    score=scored_point.score,
                    conversation=conversation_data,
                    ticket_id=ticket_id_str
                )
            )
            seen_texts.add(text_content)

    # Now, perform AI analysis for Zendesk tickets that have conversations, concurrently
    analysis_tasks = []
    for detail in raw_context_details:
        if detail.source_collection.lower() == "zendesk" and detail.conversation and detail.ticket_id:
            # Create a task for each analysis
            task = analyze_ticket_conversation_with_ai(
                detail.ticket_id,
                detail.conversation,
                generative_model_name_for_analysis # Use the model specified in the request
            )
            analysis_tasks.append(task)
        else:
            # Add a placeholder for non-analyzed items to keep order
            analysis_tasks.append(None)

    # Run all analysis tasks concurrently
    if any(task is not None for task in analysis_tasks): # Only run gather if there are actual tasks
        print(f"Starting AI analysis for {sum(1 for task in analysis_tasks if task is not None)} Zendesk tickets...")
        ai_solutions_results = await asyncio.gather(*(task for task in analysis_tasks if task is not None)) # Filter out Nones before gather

        # Assign results back to the context details
        result_idx = 0
        for i, detail in enumerate(raw_context_details):
            if analysis_tasks[i] is not None: # This context item was scheduled for analysis
                detail.ai_generated_solutions = ai_solutions_results[result_idx]
                result_idx += 1
    else:
        print("No Zendesk tickets with conversations found for AI analysis.")


    return raw_context_details


def generate_answer_with_context(
    question: str,
    context_details: List[ContextDetail],
    generative_model_id: str
) -> str:
    if not context_details:
        prompt = f"Answer the following question. You have no specific context.\nQuestion: {question}\nAnswer:"
    else:
        context_items = []
        for i, cd in enumerate(context_details):
            item_str = f"Context Snippet {i+1} from '{cd.source_collection}' (Score: {cd.score:.4f}):\n{cd.text}"
            if cd.ticket_id:
                item_str += f"\n(Related Ticket ID: {cd.ticket_id})"
            
            # Include conversation summary if present
            if cd.conversation:
                convo_summary = "\nConversation Summary (first few messages):\n"
                for c_idx, comment in enumerate(cd.conversation[:2]): # Show first 2 for brevity
                    author_name = comment.get('author_name', f"User {comment.get('author_id', 'Unknown')}")
                    comment_body = comment.get('plain_body', comment.get('body', 'No content')).strip()
                    convo_summary += f"  - {author_name}: {comment_body[:100]}...\n"
                item_str += convo_summary

            # Include AI-generated solutions if present
            if cd.ai_generated_solutions:
                item_str += f"\nAI Suggested Solutions for Ticket {cd.ticket_id}:\n{cd.ai_generated_solutions}\n"
            
            context_items.append(item_str)

        context_str = "\n\n---\n\n".join(context_items)
        prompt = f"""You are a helpful AI assistant. Answer the user's question based *only* on the provided context.
The context may include text snippets, summaries of ticket conversations, and AI-suggested general solutions derived from analyzing past tickets.
Synthesize all this information to provide a comprehensive answer.
If AI-suggested general solutions (derived from a specific ticket) are provided, consider how these general solutions might help answer the current user's question, especially if the problems seem related.
If the context doesn't contain the answer, clearly state that the information is not available.
Do not make up information.

Provided Context:
---
{context_str}
---

Question: {question}

Answer:"""

    try:
        model_to_use = genai.GenerativeModel(model_name=generative_model_id)
        print(f"\n--- PROMPT FOR FINAL ANSWER (Model: {generative_model_id}) ---")
        print(prompt) # Good for debugging
        print("--- END OF FINAL ANSWER PROMPT ---\n")
        response = model_to_use.generate_content(prompt)
        return response.text
    except Exception as e:
        # ... (error handling for LLM - keep as before) ...
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
    
    # Pass generative_model_name to search_qdrant for ticket analysis
    retrieved_context_details = await search_qdrant(
        query_embedding,
        request.collection_names,
        request.top_k,
        request.generative_model_name # Use same model for analysis
    )

    if not retrieved_context_details:
        print("No relevant context found in Qdrant.")
        # Optionally, still generate an answer without context
        # answer = generate_answer_with_context(request.question, [], request.generative_model_name)
        # return ChatResponse(answer=answer, retrieved_context=[], model_used=request.generative_model_name)

    # For debugging, print if solutions were generated
    for detail in retrieved_context_details:
        if detail.ai_generated_solutions:
            print(f"Ticket {detail.ticket_id} has AI solutions: {detail.ai_generated_solutions[:100]}...")


    answer = generate_answer_with_context(
        request.question,
        retrieved_context_details,
        request.generative_model_name
    )
    print(f"Generated final answer: {answer[:200]}...")

    return ChatResponse(
        answer=answer,
        retrieved_context=retrieved_context_details, # This will now include ai_generated_solutions
        model_used=request.generative_model_name
    )

@app.get("/")
async def root():
    # ... (root endpoint status checks - keep as before, maybe add a note about ticket analysis capability)
    q_status = "Not Connected"
    if qdrant_client:
        try: qdrant_client.get_collections(); q_status = "Connected"
        except: q_status = "Connection issue"
    g_sdk_status = "Configured" if GOOGLE_API_KEY else "Not Configured"
    es_status = "Initialized" if embedding_service else "Not Initialized"
    zd_status = "Configured & Client Initialized" if all([ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN, httpx_client]) else "Not Fully Configured"

    return {
        "message": "Dynamic Model & Multi-Collection KB Chatbot API (Azure Embeddings + Gemini Generation + Zendesk Ticket Analysis)",
        "qdrant_status": q_status,
        "google_ai_sdk_status": g_sdk_status,
        "embedding_service_status": es_status,
        "zendesk_status": zd_status,
        "features": [
            "Multi-collection Qdrant search",
            "Azure OpenAI for embeddings",
            "Google Gemini for text generation",
            "Zendesk ticket conversation retrieval",
            "AI-powered analysis of Zendesk ticket conversations for potential solutions"
        ],
        "example_generative_models_to_try": [
            "gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"
        ],
        "expected_payload_for_chat": {
            "question": "User is unable to get inapp campaigns, what could be wrong?",
            "collection_names": ["zendesk", "internal_docs"],
            "generative_model_name": "gemini-1.5-flash-latest",
            "top_k": 3
        }
    }