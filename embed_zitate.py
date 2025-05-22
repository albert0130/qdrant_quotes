import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import AzureOpenAI
from tqdm import tqdm  # ✅ Progress bar

# ✅ Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.local"))

# ✅ Init Azure OpenAI embedding client
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
embedding_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ✅ Test if deployment is valid
try:
    test = embedding_client.embeddings.create(
        model=embedding_deployment,
        input=["This is a test embedding."]  # Test with a batch
    )
    print("✅ Azure embedding deployment is working.")
except Exception as e:
    print("❌ Azure embedding deployment test failed.")
    print(e)
    exit(1)

# ✅ Qdrant client
qdrant = QdrantClient("http://localhost:6333")

def embed_text_batch(texts: list) -> list:
    """Generate vector embeddings for a batch of texts using Azure OpenAI."""
    try:
        response = embedding_client.embeddings.create(
            model=embedding_deployment,
            input=texts
        )

        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"❌ Error while parsing embeddings: {e}")
        return []


def run_embedding_pipeline(user_email: str):
    safe_email = user_email.replace("@", "_at_").replace(".", "_dot_")
    zitate_file = f"static/{safe_email}/zitate_{safe_email}.json"

    if not os.path.exists(zitate_file):
        print(f"❌ Datei nicht gefunden: {zitate_file}")
        return

    with open(zitate_file, "r", encoding="utf-8") as f:
        zitate = json.load(f)

    print(f"📄 {len(zitate)} Zitate geladen. Starte Embedding...")

    # Prepare batch of quotes to send in one request
    texts = [zitat.get("quote") or zitat.get("text") for zitat in zitate if zitat.get("quote") or zitat.get("text")]

    if not texts:
        print("⚠️ Keine validen Texte zum Embedding gefunden.")
        return

    # Show progress bar for embedding texts
    print("🔁 Generating embeddings...")
    embeddings = []
    for batch in tqdm(range(0, len(texts), 100), desc="Embeddings batch", unit="batch"):  # Process in batches of 100
        batch_texts = texts[batch:batch+100]
        batch_embeddings = embed_text_batch(batch_texts)
        embeddings.extend(batch_embeddings)

    points = []
    for zitat, embedding in zip(zitate, embeddings):
        payload = {
            "quote": zitat.get("quote") or zitat.get("text"),
            "category": zitat.get("category") or zitat.get("kat"),
            "subcategory": zitat.get("subcategory") or zitat.get("sub"),
            "profile": zitat.get("profile") or zitat.get("Profil", {}),
            "user_email": user_email,
            "created_at": zitat.get("created_at") or datetime.utcnow().isoformat()
        }
        points.append(
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector=embedding,
                payload=payload
            )
        )

    if not points:
        print("⚠️ Keine gültigen Zitate zum Einfügen vorhanden.")
        return

    # ✅ Create collection if needed
    if not qdrant.collection_exists("quotes"):
        qdrant.recreate_collection(
            collection_name="quotes",
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        )

    # Show progress bar for Qdrant insertion
    print("🔁 Inserting embeddings into Qdrant...")
    for _ in tqdm(range(0, len(points), 100), desc="Inserting points", unit="batch"):
        qdrant.upsert(
            collection_name="quotes",
            points=points[_:_+100]
        )

    print(f"\n✅ {len(points)} Zitate für {user_email} erfolgreich in Qdrant gespeichert.")

# 🔽 Run directly
if __name__ == "__main__":
    run_embedding_pipeline("example@example.com")
