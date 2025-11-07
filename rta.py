import anthropic
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime

class ConversationSimulator:
    """
    Simulates a conversation between two humans using Claude API.
    Includes pre/post survey capability to measure preference changes.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the simulator with API key."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"
        self.survey_results = []  # Stores all survey responses
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "api_calls": [],  # Detailed log of each API call
            "by_type": {
                "survey": {"input": 0, "output": 0},
                "conversation": {"input": 0, "output": 0}
            }
        }
        
    def create_persona_prompt(self, persona: Dict[str, str]) -> str:
        """Create a system prompt for a persona."""
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
    
    def create_survey_prompt(self, persona: Dict[str, str], survey: Dict[str, any]) -> str:
        """Create a system prompt for taking a survey."""
        prompt = f"""You are {persona.get('name', 'a person')} taking a survey.

Background: {persona.get('background', 'Not specified')}
Personality: {persona.get('personality', 'Friendly and conversational')}
Current mindset: {persona.get('mindset', 'Thoughtful and honest')}

Answer the following survey questions AS THIS CHARACTER WOULD, based on their personality and current state of mind.

CRITICAL INSTRUCTIONS:
- Answer ONLY with the letter of your choice (A, B, C, or D)
- Do not include explanations, just the letter
- Be consistent with the character's personality
- Answer authentically as this person would"""
        
        return prompt
    
    def administer_survey(
        self, 
        persona: Dict[str, str], 
        survey: Dict[str, any],
        survey_stage: str = "pre"
    ) -> Dict[str, any]:
        """
        Administer a multiple choice survey to a persona.
        
        Args:
            persona: The persona taking the survey
            survey: Dictionary containing survey questions and options
            survey_stage: "pre" or "post" to indicate timing
            
        Returns:
            Dictionary with survey results
        """
        system_prompt = self.create_survey_prompt(persona, survey)
        responses = {}
        
        print(f"\n{'='*60}")
        print(f"{survey_stage.upper()}-CONVERSATION SURVEY: {persona['name']}")
        print(f"{'='*60}\n")
        
        for q_id, question_data in survey['questions'].items():
            question_text = question_data['question']
            options = question_data['options']
            
            # Format the question with options
            formatted_question = f"{question_text}\n\n"
            for opt_key, opt_text in options.items():
                formatted_question += f"{opt_key}) {opt_text}\n"
            formatted_question += "\nAnswer with only the letter (A, B, C, or D):"
            
            # Get response from Claude
            message = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": formatted_question
                }]
            )
            
            # Track token usage
            self._track_token_usage(message, "survey", persona['name'], q_id)
            
            answer = message.content[0].text.strip().upper()
            # Clean up response to just get the letter
            if len(answer) > 1:
                answer = answer[0] if answer[0] in ['A', 'B', 'C', 'D'] else answer
            
            responses[q_id] = {
                "question": question_text,
                "answer": answer,
                "answer_text": options.get(answer, "Invalid response")
            }
            
            print(f"Q: {question_text}")
            print(f"A: {answer}) {options.get(answer, 'Invalid response')}\n")
        
        # Store results in data structure
        survey_result = {
            "timestamp": datetime.now().isoformat(),
            "persona_name": persona['name'],
            "stage": survey_stage,
            "responses": responses
        }
        
        self.survey_results.append(survey_result)
        
        return survey_result
    
    def get_response(self, persona: Dict[str, str], conversation_history: List[Dict[str, str]]) -> str:
        """Get a response from Claude for a specific persona."""
        system_prompt = self.create_persona_prompt(persona)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history
        )
        
        # Track token usage
        self._track_token_usage(response, "conversation", persona['name'])
        
        return response.content[0].text
    
    def _track_token_usage(self, message, call_type: str, persona_name: str, question_id: str = None):
        """
        Track token usage from API responses.
        
        Args:
            message: The API response object
            call_type: "survey" or "conversation"
            persona_name: Name of the persona
            question_id: Optional question ID for survey calls
        """
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        total = input_tokens + output_tokens
        
        # Update totals
        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["total_tokens"] += total
        
        # Update by type
        self.token_usage["by_type"][call_type]["input"] += input_tokens
        self.token_usage["by_type"][call_type]["output"] += output_tokens
        
        # Log individual call
        call_log = {
            "timestamp": datetime.now().isoformat(),
            "type": call_type,
            "persona": persona_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total
        }
        
        if question_id:
            call_log["question_id"] = question_id
        
        self.token_usage["api_calls"].append(call_log)
    
    def run_conversation_with_survey(
        self,
        persona_a: Dict[str, str],
        persona_b: Dict[str, str],
        survey: Dict[str, any],
        surveyed_persona: str,  # "a" or "b"
        initial_message: str,
        num_turns: int = 5,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run a conversation with pre and post surveys for one participant.
        
        Args:
            persona_a: First persona configuration
            persona_b: Second persona configuration
            survey: Survey configuration with questions
            surveyed_persona: Which persona takes the survey ("a" or "b")
            initial_message: Starting message
            num_turns: Number of exchanges
            verbose: Whether to print in real-time
            
        Returns:
            Dictionary with conversation log and survey results
        """
        # Determine which persona is being surveyed
        surveyed = persona_a if surveyed_persona == "a" else persona_b
        other = persona_b if surveyed_persona == "a" else persona_a
        
        # Administer pre-survey
        pre_survey_results = self.administer_survey(surveyed, survey, "pre")
        
        # Run conversation
        conversation_a = []
        conversation_b = []
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
        
        # Administer post-survey
        post_survey_results = self.administer_survey(surveyed, survey, "post")
        
        # Compile results
        results = {
            "conversation": full_log,
            "pre_survey": pre_survey_results,
            "post_survey": post_survey_results,
            "surveyed_persona": surveyed['name']
        }
        
        return results
    
    def analyze_survey_changes(self, pre_results: Dict, post_results: Dict) -> Dict[str, any]:
        """
        Analyze changes between pre and post survey results.
        
        Returns:
            Dictionary with change analysis
        """
        changes = {
            "total_questions": len(pre_results['responses']),
            "changed_answers": 0,
            "unchanged_answers": 0,
            "details": []
        }
        
        for q_id in pre_results['responses'].keys():
            pre_answer = pre_results['responses'][q_id]['answer']
            post_answer = post_results['responses'][q_id]['answer']
            
            changed = pre_answer != post_answer
            if changed:
                changes['changed_answers'] += 1
            else:
                changes['unchanged_answers'] += 1
            
            changes['details'].append({
                "question_id": q_id,
                "question": pre_results['responses'][q_id]['question'],
                "pre_answer": pre_answer,
                "pre_answer_text": pre_results['responses'][q_id]['answer_text'],
                "post_answer": post_answer,
                "post_answer_text": post_results['responses'][q_id]['answer_text'],
                "changed": changed
            })
        
        return changes
    
    def export_results(self, results: Dict[str, any], filename: str = None):
        """Export conversation and survey results to JSON file."""
        if filename is None:
            filename = f"conversation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults exported to: {filename}")
    
    def get_all_survey_results(self) -> List[Dict]:
        """Return all stored survey results."""
        return self.survey_results
    
    def get_token_usage_summary(self) -> Dict:
        """Return a summary of token usage."""
        return {
            "total_input_tokens": self.token_usage["total_input_tokens"],
            "total_output_tokens": self.token_usage["total_output_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
            "by_type": self.token_usage["by_type"],
            "total_api_calls": len(self.token_usage["api_calls"])
        }
    
    def get_detailed_token_usage(self) -> Dict:
        """Return detailed token usage including all API calls."""
        return self.token_usage
    
    def calculate_estimated_cost(self) -> Dict:
        """
        Calculate estimated cost based on Claude Sonnet 4.5 pricing.
        $3 per million input tokens, $15 per million output tokens.
        """
        input_cost = (self.token_usage["total_input_tokens"] / 1_000_000) * 3.0
        output_cost = (self.token_usage["total_output_tokens"] / 1_000_000) * 15.0
        total_cost = input_cost + output_cost
        
        return {
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(total_cost, 4),
            "input_tokens": self.token_usage["total_input_tokens"],
            "output_tokens": self.token_usage["total_output_tokens"]
        }
    
    def reset_token_tracking(self):
        """Reset token usage tracking."""
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "api_calls": [],
            "by_type": {
                "survey": {"input": 0, "output": 0},
                "conversation": {"input": 0, "output": 0}
            }
        }


# Example usage
if __name__ == "__main__":
    # Define personas
    persona_alice = {
        "name": "Alice",
        "background": "A software engineer considering a career change to teaching",
        "personality": "Analytical, cautious, values stability",
        "style": "Thoughtful and methodical",
        "mindset": "Uncertain about big life changes",
        "goals": "Seek advice and perspective"
    }
    
    persona_bob = {
        "name": "Bob",
        "background": "A former teacher who became an entrepreneur",
        "personality": "Encouraging, risk-taker, optimistic",
        "style": "Enthusiastic and storytelling-focused",
        "goals": "Share experiences and inspire confidence"
    }
    
    # Define a survey about career change attitudes
    career_survey = {
        "title": "Career Change Attitude Survey",
        "questions": {
            "q1": {
                "question": "How do you feel about taking risks in your career?",
                "options": {
                    "A": "Very uncomfortable - I prefer stability",
                    "B": "Somewhat uncomfortable - I need careful planning",
                    "C": "Somewhat comfortable - calculated risks are okay",
                    "D": "Very comfortable - I embrace change and uncertainty"
                }
            },
            "q2": {
                "question": "How confident are you in your ability to succeed in a new field?",
                "options": {
                    "A": "Not confident - too many unknowns",
                    "B": "Slightly confident - I have doubts",
                    "C": "Moderately confident - I can learn",
                    "D": "Very confident - I trust my abilities"
                }
            },
            "q3": {
                "question": "How important is financial security compared to personal fulfillment?",
                "options": {
                    "A": "Financial security is paramount",
                    "B": "Financial security is more important",
                    "C": "Both are equally important",
                    "D": "Personal fulfillment is more important"
                }
            },
            "q4": {
                "question": "When facing a major decision, what's your approach?",
                "options": {
                    "A": "Analyze extensively, delay if uncertain",
                    "B": "Plan carefully with clear milestones",
                    "C": "Research but trust intuition",
                    "D": "Act quickly when it feels right"
                }
            }
        }
    }
    
    # Initialize simulator
    sim = ConversationSimulator(api_key="REMOVED_TOKEN")
    
    # Run conversation with survey
    print("SCENARIO: Alice seeks advice from Bob about career change")
    results = sim.run_conversation_with_survey(
        persona_a=persona_alice,
        persona_b=persona_bob,
        survey=career_survey,
        surveyed_persona="a",  # Alice takes the survey
        initial_message="I've been thinking a lot about leaving tech to become a teacher. Part of me is excited, but honestly, I'm terrified of giving up my stability.",
        num_turns=6
    )
    
    # Analyze changes
    print("\n" + "="*60)
    print("SURVEY CHANGE ANALYSIS")
    print("="*60 + "\n")
    
    changes = sim.analyze_survey_changes(results['pre_survey'], results['post_survey'])
    
    print(f"Total Questions: {changes['total_questions']}")
    print(f"Changed Answers: {changes['changed_answers']}")
    print(f"Unchanged Answers: {changes['unchanged_answers']}")
    print(f"\nChange Details:\n")
    
    for detail in changes['details']:
        status = "✓ CHANGED" if detail['changed'] else "○ No change"
        print(f"{status} - {detail['question']}")
        print(f"  Pre:  {detail['pre_answer']}) {detail['pre_answer_text']}")
        print(f"  Post: {detail['post_answer']}) {detail['post_answer_text']}\n")
    
    # Export everything
    final_results = {
        "conversation_and_surveys": results,
        "change_analysis": changes,
        "all_survey_data": sim.get_all_survey_results(),
        "token_usage": sim.get_detailed_token_usage(),
        "cost_estimate": sim.calculate_estimated_cost()
    }
    
    sim.export_results(final_results)
    
    # Print token usage summary
    print("\n" + "="*60)
    print("TOKEN USAGE SUMMARY")
    print("="*60 + "\n")
    
    summary = sim.get_token_usage_summary()
    cost = sim.calculate_estimated_cost()
    
    print(f"Total API Calls: {summary['total_api_calls']}")
    print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
    print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    print(f"\nBy Type:")
    print(f"  Survey - Input: {summary['by_type']['survey']['input']:,}, Output: {summary['by_type']['survey']['output']:,}")
    print(f"  Conversation - Input: {summary['by_type']['conversation']['input']:,}, Output: {summary['by_type']['conversation']['output']:,}")
    print(f"\nEstimated Cost:")
    print(f"  Input: ${cost['input_cost_usd']:.4f}")
    print(f"  Output: ${cost['output_cost_usd']:.4f}")
    print(f"  Total: ${cost['total_cost_usd']:.4f}")