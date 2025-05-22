import os
from openai import AzureOpenAI
from dotenv import load_dotenv

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
load_dotenv(dotenv_path=".env.local")


llm_model = os.getenv("AZURE_LLM_OPENAI_MODEL")
llm_endpoint = os.getenv("AZURE_LLM_OPENAI_ENDPOINT")
llm_api_key = os.getenv("AZURE_LLM_OPENAI_KEY")
llm_version = os.getenv("AZURE_LLM_OPENAI_VERSION")
print ("LLM Model: ", llm_model)
print ("LLM Endpoint: ", llm_endpoint)
print ("LLM API Key: ", llm_api_key)
print ("LLM Version: ", llm_version)

llm_client = AzureOpenAI(
    api_key=llm_api_key,
    api_version=llm_version,
    azure_endpoint=llm_endpoint 
)

response = llm_client.chat.completions.create(
    model=llm_model,
    messages=messages,
    temperature=0.5,
    max_tokens=2000
)
print(response.choices[0].message.content)