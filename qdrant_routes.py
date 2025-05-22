from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime, timedelta
from openai import AzureOpenAI
from qdrant_client import models, QdrantClient
from dotenv import load_dotenv
from pathlib import Path
import os

router = APIRouter()
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env.local")

qdrant = QdrantClient("http://localhost:6333")
print(os.getenv("AZURE_LLM_OPENAI_KEY"))
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_LLM_OPENAI_KEY"),
    api_version=os.getenv("AZURE_LLM_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_LLM_OPENAI_ENDPOINT")
)
embedding_deployment = os.getenv("AZURE_LLM_OPENAI_DEPLOYMENT")

llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
llm_model = os.getenv("LLM_OPENAI_MODEL")

class SearchRequest(BaseModel):
    query: str
    user_email: str
    is_standard: bool
    category: str
    zeitraum: str

@router.post("/search_quotes")
async def search_quotes(search: SearchRequest):
    query = search.query.strip()
    user_email = search.user_email
    is_standard = search.is_standard
    category = search.category
    zeitraum = search.zeitraum

    filters = [
        models.FieldCondition(key="user_email", match=models.MatchValue(value=user_email))
    ]
    if category and category.lower() != "all":
        filters.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))

    if zeitraum and zeitraum != "any":
        if zeitraum == "older":
            threshold = (datetime.utcnow() - timedelta(days=365)).isoformat()
            filters.append(models.FieldCondition(key="created_at", range=models.Range(lt=threshold)))
        else:
            days = int(zeitraum.replace("d", ""))
            threshold = (datetime.utcnow() - timedelta(days=days)).isoformat()
            filters.append(models.FieldCondition(key="created_at", range=models.Range(gte=threshold)))

    query_filter = models.Filter(must=filters)

    # ✅ Nur beim Initialstart (kein Query, kein Klick): alle Zitate
    if not query and not is_standard:
        hits, _ = qdrant.scroll(collection_name="quotes", scroll_filter=query_filter, limit=100)
        return {"quotes": [h.payload for h in hits]}

    # ❌ Kein Query, aber is_standard (Sidebar-Klick) → keine Ausgabe
    if not query:
        return {"quotes": []}

    # GPT-Verfeinerung
    refined_query = query
    if not is_standard:
        try:
            refine_response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
Du bist eine KI, die Suchanfragen von Copywritern analysiert und deren inhaltlichen Kern in einem präzisen Satz formuliert, der optimal für die Nutzung in einer semantischen Vektor-Datenbank geeignet ist.
Deine Aufgabe ist es, aus einer kurzen oder vagen Suchanfrage:
- das zentrale Thema zu erkennen und konreter machen
- es präzise, eindeutig und semantisch aussagekräftig zu formulieren
"""
                    },
                    {
                        "role": "user",
                        "content": f"Originale Anfrage: {query}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            refined_query = refine_response.choices[0].message.content.strip()
            print(f"🎯 Ursprünglich: {query}")
            print(f"✨ Verfeinert: {refined_query}")
        except Exception as e:
            print(f"⚠️ GPT-Verfeinerung fehlgeschlagen: {e}")

    query_vector = embedding_client.embeddings.create(
        model=embedding_deployment,
        input=refined_query
    ).data[0].embedding

    result_limit = 100 if is_standard else 100

    hits = qdrant.search(
        collection_name="quotes",
        query_vector=query_vector,
        limit=result_limit,
        query_filter=query_filter
    )

    messages = [
        {
            "role": "system",
            "content": """
Du bist ein KI-Assistent für Marketing und Zielgruppenanalyse. Du bekommst eine Liste von Zitat-Objekten.

Gib **nur** die Zitate zurück, die:
- thematisch zu 100 Prozent zur Suchanfrage passen
- emotional, anschaulich oder problemorientiert formuliert und eine passenden Antwort auf die Suche darstellen
- keine vagen Aussagen oder allgemeine Phrasen enthalten un genau passen sonst ignoriere die Aussagen

Format: Gültiges JSON-Array mit den vollständigen Objekten.
"""
        },
        {
            "role": "user",
            "content": f"Topic: {refined_query}\n\nQuotes:\n{json.dumps([h.payload for h in hits], indent=2)}"
        }
    ]

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.5,
        max_tokens=2000
    )

    content = response.choices[0].message.content
    if "```" in content:
        content = content.split("```json")[-1].split("```")[0].strip()

    try:
        parsed = json.loads(content)
        if not parsed or not isinstance(parsed, list):
            raise ValueError("GPT returned empty or invalid list.")
    except Exception as e:
        print(f"⚠️ GPT-JSON-Parsing fehlgeschlagen: {e}")
        parsed = [h.payload for h in hits]

    return {"quotes": parsed}