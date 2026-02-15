from typing import Dict, List
from openai import OpenAI
import os
    # TODO: Define system prompt
    # TODO: Set context in messages
    # TODO: Add chat history
    # TODO: Creaet OpenAI Client
    # TODO: Send request to OpenAI
    # TODO: Return response
    

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """
    Generate response using OpenAI with context.
    Matches the signature expected by chat.py.
    """
    
    # Check if API key is provided
    if not openai_key:
        return "Error: OpenAI API key is missing."

    try:
        # Initialize OpenAI client
        # Handle Vocareum specific base URL if the key starts with 'voc'
        base_url = None
        if openai_key.startswith("voc"):
            base_url = "https://openai.vocareum.com/v1"
            
        client = OpenAI(
            api_key=openai_key,
            base_url=base_url
        )
        
        # Define system prompt (Using your NASA Mission Commander persona)
        system_prompt = """You are a mission commander at NASA in the time of the Apollo and Challenger missions.

        Your role is to:
        - Answer questions about the missions, spacecraft, and any situations encountered during the missions.
        - Use the context provided to answer questions about the missions.
        - Use the context provided to provide guidance on troubleshooting.
        - Use the context provided to provide guidance on decisions to be made during the mission.

        You are to respond in a concise and professional manner, using plain English explanations of technical concepts.
        
        If the question is outside of the scope of the missions, spacecraft and any situations encountered during the missions, respond with:
        "I'm sorry, but I can only answer questions about the missions, spacecraft and any situations encountered during the missions."
        """
        
        # Initialize messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if available
        if context:
            messages.append({"role": "system", "content": f"Context information is below.\n----------------\n{context}\n----------------"})
            
        # Add conversation history
        # Filter to ensure we only send valid roles (system, user, assistant) and valid content
        # Note: We rely on the history passed from chat.py, not an internal list
        for msg in conversation_history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        # Return response content
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error communicating with OpenAI: {str(e)}"