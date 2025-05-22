# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from qdrant_client import QdrantClient
# from typing import List, Optional
# import json
# import os
# from main import app
# from dotenv import load_dotenv

# # Load environment variables (assuming you are using .env for sensitive info)
# load_dotenv()

# # Initialize Qdrant client
# client = QdrantClient(
#     host=os.getenv("QDRANT_HOST", "localhost"),
#     port=int(os.getenv("QDRANT_PORT", 6333)),
#     api_key=os.getenv("QDRANT_API_KEY", ""),
# )

# # Define the structure of the data
# class Subsubitem(BaseModel):
#     label: str
#     count: int
#     summary: str

# class Subitem(BaseModel):
#     label: str
#     count: int
#     summary: str
#     sub_items: Optional[List[Subsubitem]] = []

# class Common(BaseModel):
#     label: str
#     count: int
#     summary: str
#     sub_items: List[Subitem]

# # Function to fetch data from Qdrant for a specific user based on email
# def fetch_data_from_qdrant(user_email: str):
#     # Example query, replace with actual vector and filter for your use case
#     query_vector = [0.1, 0.2, 0.3, 0.4]  # Replace with actual vector
#     filter = {"user_email": user_email}  # Assuming the Qdrant collection has user_email in payload

#     # Perform the search with the Qdrant client
#     response = client.search(
#         collection_name="your_collection_name",  # Replace with your Qdrant collection name
#         query_vector=query_vector,
#         filter=filter,
#         limit=10  # Adjust based on your needs
#     )

#     commons = []

#     # Process search results
#     for hit in response['result']:
#         # Extract information from Qdrant hit and map it to your structure
#         common_label = hit['payload'].get('label', 'Unknown')
#         common_count = hit['payload'].get('count', 0)
#         common_summary = hit['payload'].get('summary', 'No summary available')

#         sub_items = []
#         if 'sub_items' in hit['payload']:
#             for subitem_hit in hit['payload']['sub_items']:
#                 subitem_label = subitem_hit.get('label', 'Unknown')
#                 subitem_count = subitem_hit.get('count', 0)
#                 subitem_summary = subitem_hit.get('summary', 'No summary available')
                
#                 subsub_items = []
#                 if 'subsub_items' in subitem_hit:
#                     for subsubitem_hit in subitem_hit['subsub_items']:
#                         subsub_label = subsubitem_hit.get('label', 'Unknown')
#                         subsub_count = subsubitem_hit.get('count', 0)
#                         subsub_summary = subsubitem_hit.get('summary', 'No summary available')
#                         subsub_items.append(Subsubitem(label=subsub_label, count=subsub_count, summary=subsub_summary))
                
#                 sub_items.append(Subitem(label=subitem_label, count=subitem_count, summary=subitem_summary, sub_items=subsub_items))
        
#         commons.append(Common(label=common_label, count=common_count, summary=common_summary, sub_items=sub_items))

#     return commons

# # API endpoint for analytics, accepts user email to fetch filtered data
# @app.post("/analytics")
# async def run_analyse_task(request: Request):
#     # Retrieve the user email from the request (ensure frontend sends email)
#     body = await request.json()
#     user_email = body.get("user_email", "")  # Fetch user_email from the frontend request

#     if not user_email:
#         return {"error": "No user_email provided"}
    
#     # Fetch the data from Qdrant using the email
#     commons = fetch_data_from_qdrant(user_email)
    
#     # Return the commons data
#     return {"commons": commons}
