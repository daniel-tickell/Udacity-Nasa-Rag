from typing import Dict, List
from openai import OpenAI
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_response(openai_key, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """
    Generate response using OpenAI with context.
    Matches the signature expected by chat.py.

    Args:
        user_message: User's message
        context: Context information
        conversation_history: Conversation history
        model: OpenAI model to use (default: gpt-3.5-turbo)
    """
    
    # Set the API key from environment variable
    if not openai_key:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = openai_key

    # Try block for error handling
    try:
        # Initialize OpenAI client
        # Handle Vocareum specific base URL if the key starts with 'voc'
        base_url = None
        if api_key.startswith("voc"):
            base_url = "https://openai.vocareum.com/v1"
            
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Define system prompt (Using a NASA Mission Commander persona)
        system_prompt = """You are a mission commander at NASA in the time of the Apollo and Challenger missions.

        Your role is to:
        - Answer questions about the missions, spacecraft, and any situations encountered during the missions.
        - Provide explicit citations to the source data in responses.
        - Use the context provided to answer questions about the missions and the spacecraft.
        - Use the context provided to provide guidance on troubleshooting.
        - Use the context provided to provide guidance on decisions to be made during the mission.

        You are to respond in a concise and professional manner, using plain English explanations of technical concepts.
        
        If the question is outside of the scope of the missions, spacecraft and any situations encountered during the missions, respond with:
        "I'm sorry, but I can only answer questions about the missions, spacecraft and any situations encountered during the missions."

        If the context is missing or insufficient respond with:
        "I'm sorry, but I don't know based on the provided context"
        """
        
        # Initialize messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Set context (if available)
        if context:
            messages.append({"role": "system", "content": f"Context information is below.\n----------------\n{context}\n----------------"})
            
        # Add conversation history
        # Filter to ensure we only send valid roles (system, user, assistant) and valid content
        # Limit history to prevent token overflow (e.g., last 5 messages to keep context focused)
        # Updated based on feedback from submission 1
        # Limit history to last 5 messages
        history_limit = 5
        # Check if conversation history is relevant
        relevant_history = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history
        
        for msg in relevant_history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # Using 0.7 as a balance between Conservative and Creative
            temperature=0.7,
            # Limit the response to 300 tokens
            max_tokens=300
        )
        
        # Return the response
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error communicating with OpenAI: {str(e)}"