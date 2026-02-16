from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(),
        Faithfulness()
    ]
    
    # Evaluate the response using the metrics
    # Try Block for error handling
    try:
        results = evaluate(
            dataset=[sample],
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
   
        # Return the evaluation results
        return {k: float(v) for k, v in results.items()}
    except Exception as e:
        return {"error": f"RAGAS evaluation failed: {str(e)}"}