import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Identify directories that look like ChromaDB stores (e.g., 'chroma_db', '*_db')
    potential_dirs = [d for d in current_dir.iterdir() if d.is_dir() and ('chroma' in d.name or 'db' in d.name)]

    for chroma_dir in potential_dirs:
        # Use a try block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(chroma_dir))
            
            # Retrieve list of available collections from the database
            collections = client.list_collections()
            
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}/{collection.name}"
                
                # Get document count with fallback
                try:
                    count = collection.count()
                except:
                    count = "Unknown"

                # Build information dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({chroma_dir.name})",
                    "count": count
                }
        
        except Exception as e:
            # Handle connection or access errors gracefully
            print(f"Skipping directory {chroma_dir}: {e}") # logging The skipped Directory
            pass

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    # Create a chromadb persistent client
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Return the collection with the collection_name
    try:
        # Define embedding function
        openai_ef = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        collection = client.get_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Check if filter parameter exists and is not set to "all"
    where_filter = None
    if mission_filter and mission_filter.lower() != "all":
        # Create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}

    # Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["SOURCE DOCUMENTS:\n"]

    # Loop through paired documents and their metadata using enumeration
    # As we are only sending one query, we look at the first element.
    if isinstance(documents[0], list):
        docs = documents[0]
        metas = metadatas[0] if metadatas else []
    else:
        docs = documents
        metas = metadatas

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        # Extract mission information
        mission = meta.get('mission', 'Unknown Mission').replace('_', ' ').title()
        
        # Extract source information
        source = meta.get('source', 'Unknown Source')
        
        # Extract category information
        category = meta.get('document_category', 'General').replace('_', ' ').title()
        
        # Create formatted source header
        header = f"--- Document {i+1} [Mission: {mission} | Source: {source} | Category: {category}] ---"
        context_parts.append(header)
        
        # Check document length and truncate if necessary (e.g., limit to 2000 chars)
        content = doc if len(doc) < 2000 else doc[:2000] + "... (truncated)"
        context_parts.append(content)
        context_parts.append("") # Add a newline spacing

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)