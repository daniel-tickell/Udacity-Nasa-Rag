import os
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('ragas.prompt.pydantic_prompt').setLevel(logging.ERROR)
import chromadb
from chromadb.config import Settings
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    BleuScore,  
    RougeScore,
    AnswerRelevancy, 
    Faithfulness, 
    ContextRecall, 
    ContextPrecision
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
from typing import List, Dict
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm

# Configuration settings
COLLECTION_NAME = "demo_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
#initialize openai api key
OPENAI_API_KEY = None
N_RESULTS = 3

def get_evaluator_llm():
    """Returns the evaluator LLM if API key is present."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return LangchainLLMWrapper(ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo"))
    return None

def get_evaluator_embeddings():
    """Returns the evaluator embeddings if API key is present."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small"))
    return None

def get_default_metrics():
    """Returns default metrics for evaluation."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        evaluator_llm = get_evaluator_llm()
        evaluator_embeddings = get_evaluator_embeddings()
        return [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            ContextRecall(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            BleuScore(),
            RougeScore()
        ]
    return []

# Set OpenAI API key for global default execution (if imported, this might not apply immediately)
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    print("🔑 OpenAI API key configured")
else:
    print("⚠️  No OpenAI API key - some evaluation features will be limited")

print("⚙️ Configuration set!")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Create or get collection
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ Loaded existing collection: {COLLECTION_NAME}")
except:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Demo collection for RAG evaluation"}
    )
    print(f"✅ Created new collection: {COLLECTION_NAME}")

print(f"📊 Current collection size: {collection.count()} documents")


print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Embedding model loaded!")

# Sample documents about Space Missions
sample_documents = [
    "Apollo 11 was the first mission to land humans on the Moon, achieving this milestone on July 20, 1969. The three astronauts were Neil Armstrong, Buzz Aldrin, and Michael Collins.",
    "Six crewed Apollo missions successfully landed on the Moon: Apollo 11, 12, 14, 15, 16, and 17. The program returned a total of 842 pounds (382 kilograms) of lunar rocks and soil to Earth.",
    "The Apollo 13 lunar landing was aborted due to an electrical short circuit that caused an oxygen tank in the service module to explode, disabling the primary life support and power systems.",
    "Eugene (Gene) Cernan was the last astronaut to walk on the Moon, doing so during the Apollo 17 mission in December 1972.",
    "Apollo 8 was the first crewed spacecraft to leave low Earth orbit, travel to the Moon, orbit it, and return safely to Earth.",
    "The Saturn V heavy-lift rocket was used to launch all the crewed Apollo missions to the Moon.",
    "Apollo 1 was canceled because a cabin fire erupted during a prelaunch rehearsal test on January 27, 1967, resulting in the deaths of astronauts Virgil Grissom, Edward White, and Roger Chaffee.",
    "The Lunar Roving Vehicle was first used to explore the lunar surface during the Apollo 15 mission.",
    "The Space Shuttle Challenger disaster occurred on January 28, 1986, when the shuttle broke apart 73 seconds into its flight, killing all seven crew members aboard.",
    "The Challenger tragedy was caused by the failure of O-ring seals in the right solid rocket booster, which allowed hot gas to escape and damage the external fuel tank."
]

print(f"📚 Prepared {len(sample_documents)} sample documents")

print("🔄 Adding documents to ChromaDB...")

# Generate embeddings for documents
print("Creating embeddings...")
embeddings = []
for doc in tqdm(sample_documents, desc="Generating embeddings"):
    embedding = embedding_model.encode([doc])[0].tolist()
    embeddings.append(embedding)

# Create metadata
metadatas = [{"source": f"doc_{i}", "type": "space_mission_info"} for i in range(len(sample_documents))]

# Generate IDs
existing_count = collection.count()
ids = [f"doc_{existing_count + i}" for i in range(len(sample_documents))]

# Add to collection
collection.add(
    documents=sample_documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Added {len(sample_documents)} documents to collection")
print(f"📊 Total documents in collection: {collection.count()}")


def retrieve_documents_from_collection(collection, embedding_model, query: str, n_results: int = N_RESULTS):
    """Retrieve relevant documents for a query specific to a collection"""
    # Generate query embedding
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Search collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    return {
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'distances': results['distances'][0]
    }

# The global retrieve function has been updated to accept args for encapsulation or use the global ones if needed
def retrieve_documents(query: str, n_results: int = N_RESULTS):
    """Retrieve relevant documents for a query"""
    global collection, embedding_model
    return retrieve_documents_from_collection(collection, embedding_model, query, n_results)

def generate_answer(query: str, context_docs: List[str]) -> str:
    """Generate answer using retrieved context"""
    context = "\n\n".join(context_docs)
    
    prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            # Using OpenAI
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback answer
            return f"Based on the retrieved context, here's information about {query}: " \
                   f"[Using simplified answer generation - add your OpenAI API key for better responses]"
    except Exception as e:
        return f"Error generating answer: {str(e)}"

print("✅ Answer generation function ready!")

def run_batch_evaluation(metrics_to_run=None):
    """
    Runs the batch evaluation using the provided metrics.
    If metrics_to_run is None, runs default metrics.
    """
    import json
    print("\n" + "="*50)
    print("🚀 STARTING BATCH EVALUATION")
    print("="*50)
    
    # Define ground truth answers for evaluation from existing json file
    dataset_path = "../test_questions.json"
    if not os.path.exists(dataset_path):
        dataset_path = "test_questions.json"
    
    try:
        with open(dataset_path, "r") as f:
            ground_truth_data = json.load(f)
        print(f"📚 Loaded {len(ground_truth_data)} questions from test_questions.json")
    except Exception as e:
        print(f"Error loading dataset from test_questions.json: {e}")
        ground_truth_data = []
    
    print("📊 Creating evaluation dataset...")
    
    # Create evaluation dataset
    eval_data = []
    
    # We need to initialize the collection and model if they haven't been already
    # For safety in imported context:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    local_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    local_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    for gt in tqdm(ground_truth_data, desc="Processing QA Pairs", leave=False):
        question = gt.get("question", "")
        ground_truth = gt.get("answer", "")
        
        # Get retrieval results using local collection
        retrieval_results = retrieve_documents_from_collection(local_collection, local_embedding_model, question)
        
        # Generate answer
        answer = generate_answer(question, retrieval_results['documents'])
        
        eval_data.append({
            'user_input': question,
            'response': answer,
            'retrieved_contexts': retrieval_results['documents'],
            'reference': ground_truth
        })
    
    # Convert to RAGAS dataset format
    eval_df = pd.DataFrame(eval_data)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    print(f"✅ Created evaluation dataset with {len(eval_dataset)} examples")
    
    print("📈 Running RAGAS evaluation...")
    
    # Setup Metrics
    actual_metrics = get_default_metrics() if metrics_to_run is None else metrics_to_run
    
    if not actual_metrics:
         print("⚠️ No metrics selected or OpenAI API keys missing, skipping evaluation.")
         return {}
         
    try:
        print("⏳ This may take a few minutes...")
        evaluation_results = evaluate(
            dataset=eval_dataset,
            metrics=actual_metrics,
            raise_exceptions=False  # Continue evaluation even if some records fail
        )
        
        # Convert to pandas dataframe to safely iterate over columns
        df = evaluation_results.to_pandas()
        
        # Extract aggregated scores
        metric_names = [m.name for m in actual_metrics]
        aggregate_scores = {}
        
        print("\n🎉 Evaluation Results:")
        for metric_name in metric_names:
            # Special handling for RougeScore which is renamed in the df
            col_name = metric_name
            if metric_name == "rouge_score" and "rouge_score(mode=fmeasure)" in df.columns:
                col_name = "rouge_score(mode=fmeasure)"
                
            if col_name in df.columns:
                scores = df[col_name].tolist()
                # Filter out None/NaN
                valid_scores = [s for s in scores if s is not None and not pd.isna(s)]
                avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
                aggregate_scores[metric_name] = avg_score
                print(f"{metric_name}: {avg_score:.3f}")
            else:
                 print(f"⚠️ Metric '{metric_name}' not found in result columns.")
                
        return aggregate_scores
        
    except Exception as e:
        print(f"⚠️ Evaluation Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    
if __name__ == "__main__":
    # Test retrieval with sample queries (Moved from global scope)
    test_queries = [
        "Apollo 11 mission",
        "Challenger disaster",
        "Lunar Roving Vehicle"
    ]
    
    print("🔍 Testing document retrieval...")
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = retrieve_documents(query, n_results=2)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'], results['metadatas'], results['distances']
        )):
            similarity = 1 - distance
            print(f"  {i+1}. [Similarity: {similarity:.3f}] {doc[:100]}...")

    # Sample questions for testing (Moved from global scope)
    sample_questions = [
        "Which Apollo mission first landed on the Moon?",
        "What caused the Apollo 13 mission to abort?",
        "When did the Challenger disaster occur?"
    ]
    
    print("🤖 Running interactive Q&A session...")
    
    qa_results = []
    for question in sample_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        
        # Retrieve relevant documents
        retrieval_results = retrieve_documents(question)
        
        print(f"\n🔍 Retrieved {len(retrieval_results['documents'])} relevant documents:")
        for i, doc in enumerate(retrieval_results['documents']):
            print(f"  {i+1}. {doc[:80]}...")
        
        # Generate answer
        answer = generate_answer(question, retrieval_results['documents'])
        
        print(f"\n🎯 Generated Answer:")
        print(f"   {answer}")
        
        # Store for evaluation
        qa_results.append({
            'question': question,
            'answer': answer,
            'contexts': retrieval_results['documents']
        })
    
    print(f"\n✅ Completed {len(qa_results)} Q&A interactions")

    # Run the batch evaluation directly if executed as main
    results = run_batch_evaluation(get_default_metrics())
    print("\nFinal Results:", results)