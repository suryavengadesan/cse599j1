import anthropic
import os
from typing import List, Dict

class ConversationSimulator:
    """
    Simulates a conversation between two humans using Claude API.
    Supports customizable personas and conversation parameters.
    """
    #REMOVED_TOKEN
    def __init__(self, api_key: str = None):
        """Initialize the simulator with API key."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"
        
    def create_persona_prompt(self, persona: Dict[str, str]) -> str:
        """
        Create a system prompt for a persona.
        
        Args:
            persona: Dictionary with keys like 'name', 'background', 'personality', 'goals'
        """
        prompt = f"""You are roleplaying as {persona.get('name', 'a person')}.

Background: {persona.get('background', 'Not specified')}
Personality: {persona.get('personality', 'Friendly and conversational')}
Communication Style: {persona.get('style', 'Natural and authentic')}
Goals in Conversation: {persona.get('goals', 'Engage meaningfully')}

IMPORTANT: 
- Respond naturally as this character would
- Keep responses concise (2-4 sentences typical)
- Show personality through word choice and tone
- React authentically to what the other person says
- Don't break character or mention you're an AI"""
        
        return prompt
    
    def get_response(self, persona: Dict[str, str], conversation_history: List[Dict[str, str]]) -> str:
        """
        Get a response from Claude for a specific persona.
        
        Args:
            persona: The persona configuration
            conversation_history: List of previous messages
        """
        system_prompt = self.create_persona_prompt(persona)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history
        )
        
        return response.content[0].text
    
    def run_conversation(
        self,
        persona_a: Dict[str, str],
        persona_b: Dict[str, str],
        initial_message: str,
        num_turns: int = 5,
        verbose: bool = True
    ) -> List[Dict[str, str]]:
        """
        Run a conversation between two personas.
        
        Args:
            persona_a: First persona configuration
            persona_b: Second persona configuration
            initial_message: Starting message from persona A
            num_turns: Number of back-and-forth exchanges
            verbose: Whether to print conversation in real-time
        
        Returns:
            Full conversation history
        """
        conversation_a = []  # Persona A's view of conversation
        conversation_b = []  # Persona B's view of conversation
        full_log = []
        
        current_message = initial_message
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CONVERSATION: {persona_a['name']} & {persona_b['name']}")
            print(f"{'='*60}\n")
            print(f"{persona_a['name']}: {initial_message}\n")
        
        full_log.append({
            "speaker": persona_a['name'],
            "message": initial_message
        })
        
        for turn in range(num_turns):
            # Persona B responds
            conversation_b.append({
                "role": "user",
                "content": current_message
            })
            
            response_b = self.get_response(persona_b, conversation_b)
            
            conversation_b.append({
                "role": "assistant",
                "content": response_b
            })
            
            if verbose:
                print(f"{persona_b['name']}: {response_b}\n")
            
            full_log.append({
                "speaker": persona_b['name'],
                "message": response_b
            })
            
            # Check if we've completed all turns
            if turn == num_turns - 1:
                break
            
            # Persona A responds
            conversation_a.append({
                "role": "user",
                "content": response_b
            })
            
            response_a = self.get_response(persona_a, conversation_a)
            
            conversation_a.append({
                "role": "assistant",
                "content": response_a
            })
            
            if verbose:
                print(f"{persona_a['name']}: {response_a}\n")
            
            full_log.append({
                "speaker": persona_a['name'],
                "message": response_a
            })
            
            current_message = response_a
        
        if verbose:
            print(f"{'='*60}\n")
        
        return full_log


# Example usage
if __name__ == "__main__":
    # Define two personas
    persona_alice = {
        "name": "Alice",
        "background": "A software engineer who loves hiking and photography",
        "personality": "Enthusiastic, curious, and analytical",
        "style": "Uses technical language naturally, often relates things to code",
        "goals": "Share experiences and learn about others"
    }
    
    persona_bob = {
        "name": "Bob",
        "background": "A chef who recently opened a farm-to-table restaurant",
        "personality": "Warm, passionate about food, enjoys storytelling",
        "style": "Descriptive, often uses food metaphors",
        "goals": "Connect over shared interests and tell engaging stories"
    }
    
    # Alternative personas for different scenarios
    persona_skeptic = {
        "name": "Jordan", # "Carol"
        "background": "A philosophy professor specializing in epistemology",
        "personality": "Questioning, thoughtful, plays devil's advocate",
        "style": "Socratic, asks probing questions",
        "goals": "Understand deeper implications and challenge assumptions"
    }
    
    persona_optimist = {
        "name": "Sam",
        "background": "A motivational coach and community organizer",
        "personality": "Positive, energetic, sees opportunities everywhere",
        "style": "Encouraging, uses affirmations and forward-looking language",
        "goals": "Inspire and find common ground"
    }
    
    # Initialize simulator
    sim = ConversationSimulator(api_key="REMOVED_TOKEN")
    
    # Run a conversation
    print("SCENARIO 1: Alice and Bob discuss weekend plans")
    conversation = sim.run_conversation(
        persona_a=persona_alice,
        persona_b=persona_bob,
        initial_message="Hey! I'm thinking about trying something new this weekend. Any suggestions?",
        num_turns=6
    )
    
    # Try a different pairing
    print("\n\nSCENARIO 2: Jordan and Sam discuss career changes")
    conversation2 = sim.run_conversation(
        persona_a=persona_skeptic,
        persona_b=persona_optimist,
        initial_message="I've been thinking about making a major career change, but I'm not sure if it's the right move.",
        num_turns=5
    )
    
    # Export conversation to file (optional)
    import json
    with open("conversation_log.json", "w") as f:
        json.dump({
            "scenario_1": conversation,
            "scenario_2": conversation2
        }, f, indent=2)