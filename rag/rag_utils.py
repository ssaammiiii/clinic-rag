
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb 
from chromadb.config import Settings
from openai import AzureOpenAI


import utils.data_utils
import utils.api_utils
from dotenv import load_dotenv
from chromadb import PersistentClient
from utils.api_utils import fetch_semantic_scholar
from utils.data_utils import search_patient


# --- Initialize Chroma ---
client = PersistentClient(path="./chroma_db")
collection_name = "medical_papers"
try:
    collection = client.get_collection(name=collection_name)
except Exception:
    collection = client.create_collection(name=collection_name)

# --- Azure OpenAI client ---
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_MODEL"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]

llm_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2025-01-01-preview"
)

# --- Embedding function ---
embedding_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

def get_embedding(text):
    result = llm_client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return result.data[0].embedding


# --- Add papers to Chroma (Optimized Version) ---
def add_papers_to_chroma(papers):
    
    # Prepare lists for batch adding
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    embeddings_to_add = []

    for paper in papers:
        # --- Data validation ---
        abstract = paper.get("abstract")
        title = paper.get("title")
        paper_id = paper.get("paperId") # <-- Use the unique paperId
        year = paper.get("year") # *** CHANGE 1A: EXTRACTING THE YEAR ***


        # Skip the paper if key info is missing
        if not abstract or not paper_id or not title or not year:
            print(f"Skipping paper, missing data. Title: {title}")
            continue

        # --- Get embedding (handle potential errors) ---
        try:
            doc_content = f"Title: {title}\nAbstract: {abstract}" #embedding both title and abstract
            emb = get_embedding(doc_content)
            
            # Add to our batch lists
            documents_to_add.append(doc_content)
            metadatas_to_add.append({
                "title": title, 
                "url": paper.get("url", ""),
                "year": year  # *** CHANGE 1B: ADDING YEAR TO METADATA ***  
            })

            ids_to_add.append(paper_id) # Use the unique ID
            embeddings_to_add.append(emb)

        except Exception as e:
            print(f"Error embedding paper {paper_id} ('{title}'): {e}")

    # --- Add all papers in ONE batch ---
    if ids_to_add:
        collection.add(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add,
            embeddings=embeddings_to_add
        )
        
        print(f"Successfully added {len(ids_to_add)} new papers to Chroma.")
    else:
        print("No new valid papers to add.")


# --- Doctor query ---
def ask_doctor_chat(query, patient_name=None, top_k=5):
    context = ""
    if patient_name:
        patient = search_patient(patient_name)
        if patient:
            context += f"Patient info: {patient}\n"

    # --- Step 1: Query Chroma for related papers ---
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]



    # --- Step 2: Check relevance and decide whether to fetch new papers ---
    should_fetch = False

    if not documents or len(documents) == 0:
        should_fetch = True
    else:
        #Creating a similarity score list from distances
        similarities = [1 - d for d in distances]
        similarity_threshold = 0.5 # Corresponds to distance of 0.35\


        # If all retrieved results have poor similarity (distance > 0.35), fetch new ones
        if all(sim < similarity_threshold for sim in similarities):
            avg_similarity = sum(similarities) / len(similarities)
            print(f"Low similarity detected (avg similarity {avg_similarity:.2f}) → fetching new papers...")
            should_fetch = True


    if should_fetch:
        print("No relevant or highly similar papers found. Fetching new ones...")
        new_papers = fetch_semantic_scholar(query, limit=3)

        if new_papers:
            add_papers_to_chroma(new_papers)
            print(f"Added {len(new_papers)} new papers to Chroma for query '{query}'.")

            # re-query after adding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents"]
            )
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
        else:
            print("No new papers found from API either.")


    # --- Step 3: Build context from the retrieved papers ---
    if documents:
        context += "Relevant research papers:\n"
        for doc, meta in zip(documents, metadatas):
            title = meta.get('title', 'Untitled')
            url = meta.get('url', 'N/A')
            year = meta.get('year', 'N/A')
            
            # *** CHANGE 2A: STRUCTURE THE CITATION METADATA EXPLICITLY ***
            context += f"SOURCE METADATA: Title: {title}, Year: {year}, URL: {url}\n"
            context += f"DOCUMENT CONTENT: {doc}\n\n"
    else:
        context += "No relevant research papers found.\n"

    # --- Step 4: Send to LLM ---
    system_instruction = (
        "You are an expert medical research assistant. Answer the user's question "
        "concisely and professionally, relying *only* on the provided context. "
        "Crucially, for every piece of information you provide, you MUST cite the source document "
        "by including its **Title**, **Publication Year**, and **URL**, as provided in the SOURCE METADATA."
    )
    
    prompt = f"Context:\n{context}\nQuestion: {query}"    
    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content
