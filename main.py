from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from qdrant_client import QdrantClient, models
from typing import List, Optional
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
from webapp.generate_content_route import router as content_router
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import Request
from pathlib import Path
from qdrant_client.models import Filter, FieldCondition, MatchValue
import traceback


from qdrant_routes import qdrant
import secrets
import dataclasses

import shutil
import uuid
import subprocess
import stripe
import sqlite3

# Eigene Module

from webapp.auth import add_user, get_user_by_email
from qdrant_routes import router as qdrant_router
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

load_dotenv(dotenv_path=".env.local")
llm_model = os.getenv("AZURE_LLM_OPENAI_MODEL")
llm_endpoint = os.getenv("AZURE_LLM_OPENAI_ENDPOINT")
llm_api_key = os.getenv("AZURE_LLM_OPENAI_KEY")
llm_version = os.getenv("AZURE_LLM_OPENAI_VERSION")
print ("LLM Model: ", llm_model)
print ("LLM Endpoint: ", llm_endpoint)
print ("LLM API Key: ", llm_api_key)
print ("LLM Version: ", llm_version)

# FastAPI App
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔓 Allow all origins — change in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


app.include_router(content_router)

# Datenmodell für Qdrant-Suche
class SearchRequest(BaseModel):
    query: str
    user_email: str
    is_standard: bool = False
    category: Optional[str] = None

#load_dotenv()  # Das lädt die .env-Datei

#load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env.local")
print("✅ .env.local geladen:", os.getenv("AZURE_LLM_OPENAI_KEY"))
# stripe.api_key = os.getenv("STRIPE_API_KEY")
# if not stripe.api_key:
#     print("❌ Kein Stripe API Key geladen – prüfe .env-Datei")
# else:
#     print("✅ Stripe API Key geladen")


task_registry = {}
executor = ThreadPoolExecutor()

# from webapp.run_analyse_task import router as analytics_router

#router = APIRouter()
app.include_router(qdrant_router)
client = QdrantClient("http://localhost:6333")


llm_client = AzureOpenAI(
    api_key=llm_api_key,
    api_version=llm_version,
    azure_endpoint=llm_endpoint 
)

# Define the structure of the data
class Subsubitem(BaseModel):
    label: str
    count: int
    summary: str

class Subitem(BaseModel):
    label: str
    count: int
    summary: str
    sub_items: Optional[List[Subsubitem]] = []

class Common(BaseModel):
    label: str
    count: int
    summary: str
    sub_items: List[Subitem]
    
class AnalyticsRequest(BaseModel):
    category: str
    user_email: str
    zeitraum: str

# Function to fetch data from Qdrant for a specific user based on email
def fetch_data_from_qdrant(user_email: str, category: Optional[str] = None, zeitraum: Optional[str] = None):
    print(f"🔍 Fetching Qdrant data for {user_email}, category={category}, zeitraum={zeitraum}")

    query_vector = [0.1] * 1536  # Adjust to your actual vector size
    print("✅ Using dummy query vector of correct size")

    filters = [
        models.FieldCondition(
            key="user_email",
            match=models.MatchValue(value=user_email)
        )
    ]
    print("User email filter applied.")

    if category and category.strip().lower() != "any":
        filters.append(
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category)
            )
        )
        print("Category filter applied.")

    if zeitraum and zeitraum.strip().lower() != "any":
        filters.append(
            models.FieldCondition(
                key="zeitraum",
                match=models.MatchValue(value=zeitraum)
            )
        )
        print("Zeitraum filter applied.")

    query_filter = models.Filter(must=filters)

    try:
        results = client.search(
            collection_name="quotes",
            query_vector=query_vector,
            query_filter=query_filter,
            limit=1000,
            with_payload=True
        )
        print(f"✅ Qdrant returned {len(results)} results")
    except Exception as e:
        print("❌ Qdrant query failed:", str(e))
        return [], 0, 0

    if not results:
        print("⚠️ No matching results from Qdrant")
        return [], 0, 0

    payloads = [hit.payload for hit in results if hit.payload]
    print(f"📊 Extracted {len(payloads)} payloads from results")

    try:
        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are an assistant that analyzes user quotes and returns structured analytics. "
        #             "Group the quotes into up to 10 commons. Each common includes label, count, summary, and sub_items. "
        #             "Each sub_item includes label, count, summary, and sub_items (optional, same structure). "
        #             "Respond in this JSON format only, and do not wrap it in ```json blocks```:\n\n"
        #             "{\n"
        #             "  \"number_of_analyzed_calls\": int,\n"
        #             "  \"number_of_extracted_quotes\": int,\n"
        #             "  \"analytics\": [\n"
        #             "    {\n"
        #             "      \"label\": str,\n"
        #             "      \"count\": int,\n"
        #             "      \"summary\": str,\n"
        #             "      \"sub_items\": [ ... ]\n"
        #             "    }\n"
        #             "  ]\n"
        #             "}"
        #         )
        #     },
        #     {
        #         "role": "user",
        #         "content": f"Quotes:\n{json.dumps(payloads, indent=2)}"
        #     }
        # ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that filters a list of quote objects based on their relevance to a given topic. "
                    "Each quote includes fields like 'quote', 'category', 'subcategory', and 'profile'. "
                    "All quotes provided already match the correct category context. "
                    "You must not discard quotes based on category — only on topic relevance. "
                    "Do not invent, modify, or fabricate quotes. Return only quote objects that are highly relevant to the topic. "
                    "Respond with a valid JSON array of full quote objects from the input."
                    "You have to answer more than 2 quotes"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Topic: problem\n\n"
                    "Return only the quotes that are highly relevant to the topic. "
                    "Respond with a valid JSON array of full quote objects."
                )
            }
        ]

        print("🤖 Sending to LLM...")


        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.5,
            max_tokens=2000
        )

        print("✅ LLM response received", response.choices[0].message.content)

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("LLM response was empty or malformed.")

        content = response.choices[0].message.content.strip()

        # ✅ Remove ```json or ``` wrappers if they exist
        if content.startswith("```json"):
            content = content.replace("```json", "").strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        analytics_json = json.loads(content)

        print("✅ LLM returned valid analytics format")

        # ✅ Build Common objects from analytics_json["analytics"]
        commons = [Common(**item) for item in analytics_json.get("analytics", [])]
        return commons, analytics_json.get("number_of_analyzed_calls", 1), analytics_json.get("number_of_extracted_quotes", len(payloads))

    except Exception as e:
        print("❌ LLM processing failed:", str(e)) # Full traceback
        traceback.print_exc()

        # Check if it's an OpenAI HTTP response error
        if hasattr(e, 'response') and e.response is not None:
            print("🔍 HTTP Status Code:", e.response.status_code)
            try:
                print("🔍 Response JSON:", e.response.json())
            except Exception:
                print("🔍 Raw response content:", e.response.text)
        elif hasattr(e, 'status_code'):
            print("🔍 Status Code:", e.status_code)
        elif hasattr(e, 'message'):
            print("🔍 Error Message:", e.message)
        else:
            print("🔍 General Exception:", str(e))

        # Return fallback values
        return [], len(set(p["quote"] for p in payloads)), len(payloads)


# Endpoint handler
@app.post("/analytics")
async def run_analyse_task(request: Request):
    try:
        body = await request.json()
        print("📥 Received request body:", body)

        data = AnalyticsRequest(**body)
        print("✅ Parsed AnalyticsRequest:", data)

        commons, analyzed_calls, extracted_quotes = fetch_data_from_qdrant(
            user_email=data.user_email,
            category=data.category,
            zeitraum=data.zeitraum
        )

        response = {
            "number_of_analyzed_calls": analyzed_calls,
            "number_of_extracted_quotes": extracted_quotes,
            "analytics": [common.dict() for common in commons]
        }

        print("📤 Returning response:", response)
        return response

    except Exception as e:
        print("❌ Error in /analytics endpoint:", str(e))
        return {
            "number_of_analyzed_calls": 0,
            "number_of_extracted_quotes": 0,
            "analytics": []
        }



@app.get("/vector-dashboard", response_class=HTMLResponse)
def show_vector_dashboard(request: Request):
    return templates.TemplateResponse("vector_dashboard.html", {"request": request})

@app.get("/vector-dashboard-updated", response_class=HTMLResponse)
def show_vector_dashboard(request: Request):
    return templates.TemplateResponse("vector_dashboard_with_sidebar_toggle.html", {"request": request})


STRIPE_PRICES = {
    "pro": "price_1RO5TfBtuM8MmsuFlDQQTbM3",
    "premium": "price_1RO5UABtuM8MmsuFLeEH3nOu"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/upload-success", response_class=HTMLResponse)
def upload_success(request: Request):
    return templates.TemplateResponse("upload_success.html", {"request": request})

@app.get("/content-generator", response_class=HTMLResponse)
async def content_generator_page():
    return FileResponse(Path(__file__).resolve().parent / "templates" / "content-generator.html")

@app.get("/pricing", response_class=HTMLResponse)
def pricing_page(request: Request):
    return templates.TemplateResponse("pricing.html", {"request": request})

@app.get("/subscribe-form", response_class=HTMLResponse)
def subscribe_form(request: Request, plan: str = "pro"):
    return templates.TemplateResponse("subscribe_form.html", {
        "request": request,
        "plan": plan
    })

@app.get("/vektor-dashboard", response_class=HTMLResponse)
def show_vector_dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Versuche, alle Zitate für den Nutzer zu laden (Dummy-Vector für unsortierte Ergebnisse)
    try:
        hits = qdrant.search(
            collection_name="quotes",
            query_vector=[0.0] * 1536,  # Dummy-Vector (gleiche Dimension wie Embedding-Modell)
            limit=20,
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="user_email",
                    match=models.MatchValue(value=user)
                )
            ])
        )
        quotes = [h.payload for h in hits]
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Zitate: {e}")
        quotes = []

    return templates.TemplateResponse("vector_dashboard.html", {
        "request": request,
        "user": user,
        "quotes": quotes
    })

@app.get("/partnership", response_class=HTMLResponse)
def partnership(request: Request):
    return templates.TemplateResponse("partnership.html", {"request": request})

@app.get("/product", response_class=HTMLResponse)
def product_page(request: Request):
    return templates.TemplateResponse("product.html", {"request": request})

@app.get("/solution", response_class=HTMLResponse)
def solution_page(request: Request):
    return templates.TemplateResponse("solution.html", {"request": request})

@app.get("/data-security", response_class=HTMLResponse)
def data_security_page(request: Request):
    return templates.TemplateResponse("data_security.html", {"request": request})

@app.get("/founder-story", response_class=HTMLResponse)
def founder_story_page(request: Request):
    return templates.TemplateResponse("founder_story.html", {"request": request})

@app.get("/support", response_class=HTMLResponse)
def support_page(request: Request):
    return templates.TemplateResponse("support.html", {"request": request})

@app.get("/abo_erforderlich", response_class=HTMLResponse)
def abo_erforderlich(request: Request):
    return templates.TemplateResponse("abo_erforderlich.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

app.include_router(qdrant_router)

@app.get("/hub", response_class=HTMLResponse)
def post_login_hub(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("post_login_hub.html", {"request": request, "user": user})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/abort-task")
def abort_task(request: Request):
    task_id = request.session.get("task_id")
    if not task_id:
        return PlainTextResponse("No task ID found.", status_code=400)

    future = task_registry.get(task_id)
    if future and not future.done():
        future.cancel()
        return PlainTextResponse("Task aborted.")
    else:
        return PlainTextResponse("Task not found or already completed.")

@app.get("/generate-consent-link", response_class=HTMLResponse)
def generate_consent_link(request: Request):
    return templates.TemplateResponse("generate_consent_link.html", {"request": request})


@app.get("/consent", response_class=HTMLResponse)
def consent_form(request: Request, redirect_url: str = "/"):
    return templates.TemplateResponse("consent.html", {"request": request, "redirect_url": redirect_url})


@app.post("/consent-submit")
async def handle_consent(email: str = Form(...), agree: str = Form(...), redirect_url: str = Form(...)):
    # Hier kannst du die Zustimmung speichern, z. B. in einer Datenbank oder Datei
    print(f"✅ Zustimmung von {email} für {redirect_url}")

    # Dann weiterleiten zum Call-Link
    return RedirectResponse(url=redirect_url, status_code=303)

# @app.post("/subscribe")
# async def subscribe(request: Request):
#     form = await request.form()
#     plan = form.get("plan")
#     email = form.get("email")

#     if plan not in STRIPE_PRICES:
#         return JSONResponse({"error": "Ungültiger Plan"}, status_code=400)

#     checkout_session = stripe.checkout.Session.create(
#         success_url="http://localhost:8000/success",
#         cancel_url="http://localhost:8000/cancel",
#         payment_method_types=["card"],
#         mode="subscription",
#         customer_email=email,
#         line_items=[{
#             "price": STRIPE_PRICES[plan],
#             "quantity": 1,
#         }]
#     )

#     return RedirectResponse(url=checkout_session.url, status_code=303)



@app.post("/register")
def register_user(
    first_name: str = Form(...),
    last_name: str = Form(...),
    address: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    if get_user_by_email(email):
        return {"error": "❌ Benutzer existiert bereits."}

    add_user(first_name, last_name, address, email, password)
    return RedirectResponse(url="/", status_code=302)

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not user.verify_password(password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "❌ Ungültige Login-Daten."})

    request.session["user"] = email
    return RedirectResponse(url="/hub", status_code=302)
@app.post("/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # 🔐 Zielordner vorbereiten
    user_safe = user.replace("@", "_at_").replace(".", "_dot_")
    upload_dir = os.path.join("uploads", user_safe)
    os.makedirs(upload_dir, exist_ok=True)

    # 📄 Eindeutiger Dateiname
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(upload_dir, filename)

    # 💾 Datei speichern
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🧠 Hintergrund-Analyse starten
    background_tasks.add_task(run_analyse_task, user, file_path)

    # ✅ Zurück zum Dashboard mit Status
    return RedirectResponse(
        url="/dashboard?analysis_started=true", 
        status_code=303
    )

from fastapi import Request, Form
from fastapi.responses import JSONResponse
from qdrant_client import models
from qdrant_routes import llm_model, qdrant
import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from qdrant_client import models
import os

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    user_email: str
    is_standard: bool
    category: str

@router.post("/search_quotes")
async def search_quotes(search: SearchRequest):
    query = search.query.strip()
    user_email = search.user_email
    is_standard = search.is_standard
    category = search.category

    # 📑 Filter
    filters = [
        models.FieldCondition(
            key="user_email",
            match=models.MatchValue(value=user_email)
        )
    ]
    if category and category.strip().lower() != "all":
        filters.append(models.FieldCondition(
            key="category",
            match=models.MatchValue(value=category)
        ))

    query_filter = models.Filter(must=filters)

    # 🧾 Kein Query → scroll
    if not query:
        hits, _ = qdrant.scroll(
            collection_name="quotes",
            scroll_filter=query_filter,
            limit=100
        )
        quotes = [h.payload for h in hits]
        return JSONResponse(content={"quotes": quotes})

    # 🔍 Embedding erzeugen
    try:
        query_vector = client.embeddings.create(
            model="text-embedding-3-small",  # dein Azure Deploymentname
            input=query
        ).data[0].embedding
    except Exception as e:
        return JSONResponse(content={"error": f"Embedding fehlgeschlagen: {str(e)}"}, status_code=500)

    # 🔎 Suche in Qdrant
    result_limit = 100 if is_standard else 20
    hits = qdrant.search(
        collection_name="quotes",
        query_vector=query_vector,
        limit=result_limit,
        query_filter=query_filter
    )

    # 🧠 LLM Re-Ranking
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that filters a list of quote objects based on their relevance to a given topic. "
                "Each quote includes 'quote', 'category', 'subcategory', 'profile', and 'user_email'. "
                "Respond with a valid JSON array of the full relevant quote objects."
            )
        },
        {
            "role": "user",
            "content": f"Topic: {query}\n\nQuotes:\n{json.dumps([h.payload for h in hits], indent=2)}"
        }
    ]

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",  # dein Azure Deploymentname für GPT
            messages=messages,
            temperature=0.5,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        quotes = json.loads(content)
        return JSONResponse(content={"quotes": quotes})

    except Exception as e:
        return JSONResponse(content={"error": f"LLM-Reranking fehlgeschlagen: {str(e)}"}, status_code=500)
# # 🟣 Direktanfrage für Buttons/Standardfragen (ohne LLM)
# @app.post("/suchanfrage")
# def suche(search: SearchRequest):
#     from webapp.qdrant_routes import client, qdrant, deployment

#     query_vector = client.embeddings.create(model=deployment, input=search.query).data[0].embedding

#     filters = [
#         models.FieldCondition(key="user_email", match=models.MatchValue(value=search.user_email))
#     ]
#     if search.category:
#         filters.append(models.FieldCondition(key="category", match=models.MatchValue(value=search.category)))

#     query_filter = models.Filter(must=filters)

#     # Standardbegriffe -> viele Treffer
#     standardbegriffe = [
#         "probleme", "schmerz", "bedürfnisse", "ziele", "ängste", "herausforderungen"
#     ]
#     limit = 100 if search.query.lower() in standardbegriffe else 20

#     hits = qdrant.search(
#         collection_name="quotes",
#         query_vector=query_vector,
#         limit=limit,
#         query_filter=query_filter
#     )

#     return JSONResponse(content={"quotes": [h.payload for h in hits]})
