import google.generativeai as genai
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
import httpx

app = FastAPI()

# --- Configuration ---

# Primary Qdrant (Zendesk)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_ZENDESK_COLLECTION = "zendesk"

# Secondary Qdrant (Support Runbook / Confluence)
QDRANT_INTERNAL_HOST = "172.174.106.224"
QDRANT_INTERNAL_PORT = 6333
QDRANT_INTERNAL_COLLECTION = "vectordbinternal"

# Global clients
qdrant_client: Optional[QdrantClient] = None
qdrant_internal_client: Optional[QdrantClient] = None
httpx_client: Optional[httpx.AsyncClient] = None

# --- Models ---


class QueryRequest(BaseModel):
    question: str
    collection_names: List[str]
    generative_model_name: str
    top_k: int = 5


class ContextDetail(BaseModel):
    text: str
    source_collection: str
    score: Optional[float] = None
    conversation: Optional[List[Dict[str, Any]]] = None
    ticket_id: Optional[str] = None
    ai_generated_solutions: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    retrieved_context: List[ContextDetail]
    model_used: str


# --- Startup and Shutdown Events ---


@app.on_event("startup")
async def startup_event():
    global qdrant_client, qdrant_internal_client, httpx_client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        qdrant_client = None

    try:
        qdrant_internal_client = QdrantClient(
            host=QDRANT_INTERNAL_HOST, port=QDRANT_INTERNAL_PORT, prefer_grpc=True
        )
        print(f"Connected to Support Runbook Qdrant at {QDRANT_INTERNAL_HOST}:{QDRANT_INTERNAL_PORT}")
    except Exception as e:
        print(f"Failed to connect to Support Runbook Qdrant: {e}")
        qdrant_internal_client = None

    httpx_client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown_event():
    global qdrant_client, qdrant_internal_client, httpx_client
    if qdrant_client:
        qdrant_client.close()
        print("Closed Qdrant client")
    if qdrant_internal_client:
        qdrant_internal_client.close()
        print("Closed Support Runbook Qdrant client")
    if httpx_client:
        await httpx_client.aclose()
        print("Closed HTTPX client")


# --- Helper Functions ---


def get_embedding(text: str) -> List[float]:
    """
    Placeholder for your embedding generation logic.
    Replace this with your actual embedding generation code.
    """
    # Example: return some vector embedding for the text
    # For demonstration, return dummy vector
    return [0.0] * 768


async def fetch_zendesk_conversation(ticket_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch conversation data for a Zendesk ticket.
    Replace with your actual API call or DB query.
    """
    # Example dummy implementation
    # Replace with real call
    return [
        {"from": "user", "message": "Example user message"},
        {"from": "agent", "message": "Example agent reply"},
    ]


async def analyze_ticket_conversation_with_ai(
    ticket_id: str, conversation: List[Dict[str, Any]], model_name: str
) -> str:
    """
    Analyze Zendesk ticket conversation using generative AI.
    """
    prompt = f"Analyze the following Zendesk ticket conversation (ID: {ticket_id}) and provide helpful solutions:\n\n"
    for turn in conversation:
        prompt += f"{turn['from'].capitalize()}: {turn['message']}\n"
    prompt += "\nSolutions:"

    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"AI analysis error for ticket {ticket_id}: {e}")
        return "No AI-generated solutions available."


async def search_qdrant(
    query_embedding: List[float],
    collection_names: List[str],
    top_k: int,
    generative_model_name_for_analysis: str,
) -> List[ContextDetail]:
    if not qdrant_client or not qdrant_internal_client:
        raise HTTPException(status_code=503, detail="Qdrant clients not available.")

    all_hits_with_collection_info: List[tuple[ScoredPoint, str]] = []

    for collection_name_str in collection_names:
        current_collection_name = collection_name_str.strip()
        if not current_collection_name:
            continue

        # Select client and actual collection name
        if current_collection_name.lower() == "confluence":
            client = qdrant_internal_client
            actual_collection_name = QDRANT_INTERNAL_COLLECTION
        elif current_collection_name.lower() == "zendesk":
            client = qdrant_client
            actual_collection_name = QDRANT_ZENDESK_COLLECTION
        else:
            # Default to primary Qdrant for unknown collections
            client = qdrant_client
            actual_collection_name = current_collection_name

        try:
            search_results = client.search(
                collection_name=actual_collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            for hit in search_results:
                all_hits_with_collection_info.append((hit, current_collection_name))
        except Exception as e:
            print(f"Warning: Error searching collection '{current_collection_name}': {e}")
            continue

    if not all_hits_with_collection_info:
        return []

    # Sort hits by score descending
    all_hits_with_collection_info.sort(key=lambda x: x[0].score, reverse=True)

    raw_context_details: List[ContextDetail] = []
    seen_texts = set()
    for scored_point, source_collection in all_hits_with_collection_info:
        if len(raw_context_details) >= top_k:
            break
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
                    ticket_id=ticket_id_str,
                )
            )
            seen_texts.add(text_content)

    # AI analysis only for Zendesk tickets with conversation
    analysis_tasks = []
    for detail in raw_context_details:
        if detail.source_collection.lower() == "zendesk" and detail.conversation and detail.ticket_id:
            task = analyze_ticket_conversation_with_ai(
                detail.ticket_id, detail.conversation, generative_model_name_for_analysis
            )
            analysis_tasks.append(task)
        else:
            analysis_tasks.append(None)

    if any(task is not None for task in analysis_tasks):
        print(f"Starting AI analysis for Zendesk tickets...")
        ai_results = await asyncio.gather(*(t for t in analysis_tasks if t is not None))
        idx = 0
        for i, detail in enumerate(raw_context_details):
            if analysis_tasks[i] is not None:
                detail.ai_generated_solutions = ai_results[idx]
                idx += 1

    return raw_context_details


def generate_answer_with_context(
    question: str, context_details: List[ContextDetail], generative_model_id: str
) -> str:
    if not context_details:
        prompt = f"Answer the following question. You have no specific context.\nQuestion: {question}\nAnswer:"
    else:
        context_items = []
        for i, cd in enumerate(context_details):
            item_str = f"Source {i+1} ({cd.source_collection}): {cd.text}"
            context_items.append(item_str)

        context_str = "\n".join(context_items)
        prompt = f"""You are a helpful AI assistant. Use only the provided sources to answer the question. Provide a concise answer and cite the sources by number.

Sources:
{context_str}

Question: {question}

Answer:"""

    try:
        model = genai.GenerativeModel(model_name=generative_model_id)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer with LLM '{generative_model_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")


# --- API Endpoint ---


@app.post("/chat", response_model=ChatResponse)
async def chat_with_kb(request: QueryRequest):
    if not request.collection_names:
        raise HTTPException(status_code=400, detail="No collection names provided.")
    if not request.generative_model_name:
        raise HTTPException(status_code=400, detail="No generative model name provided.")

    print(
        f"Received question: '{request.question}' for collections: {request.collection_names} using model: {request.generative_model_name}"
    )

    query_embedding = get_embedding(request.question)

    retrieved_context_details = await search_qdrant(
        query_embedding, request.collection_names, request.top_k, request.generative_model_name
    )

    answer_text = generate_answer_with_context(
        request.question, retrieved_context_details, request.generative_model_name
    )

    # Return only answer and minimal context (text + source collection + score)
    minimal_context = [
        ContextDetail(text=cd.text, source_collection=cd.source_collection, score=cd.score)
        for cd in retrieved_context_details
    ]

    return ChatResponse(
        answer=answer_text.strip(), retrieved_context=minimal_context, model_used=request.generative_model_name
    )
