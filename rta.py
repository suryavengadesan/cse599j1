import anthropic
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import statistics
from collections import defaultdict

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
        self.experiments = []  # Store all experiment results
        
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
    
    def run_multiple_experiments(
        self,
        persona_a: Dict[str, str],
        persona_b: Dict[str, str],
        survey: Dict[str, any],
        surveyed_persona: str,
        initial_message: str,
        num_experiments: int = 10,
        num_turns: int = 5,
        verbose: bool = False
    ) -> List[Dict[str, any]]:
        """
        Run multiple conversation experiments back-to-back.
        
        Args:
            persona_a: First persona configuration
            persona_b: Second persona configuration
            survey: Survey configuration
            surveyed_persona: Which persona takes survey ("a" or "b")
            initial_message: Starting message
            num_experiments: Number of experiments to run
            num_turns: Number of turns per conversation
            verbose: Print each conversation in detail
            
        Returns:
            List of all experiment results
        """
        print(f"\n{'='*60}")
        print(f"RUNNING {num_experiments} EXPERIMENTS")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i in range(num_experiments):
            print(f"[Experiment {i+1}/{num_experiments}] Running...")
            
            # Reset token tracking for this experiment
            experiment_start_tokens = {
                "input": self.token_usage["total_input_tokens"],
                "output": self.token_usage["total_output_tokens"]
            }
            
            # Run single experiment
            result = self.run_conversation_with_survey(
                persona_a=persona_a,
                persona_b=persona_b,
                survey=survey,
                surveyed_persona=surveyed_persona,
                initial_message=initial_message,
                num_turns=num_turns,
                verbose=verbose
            )
            
            # Calculate tokens used in this experiment
            experiment_tokens = {
                "input": self.token_usage["total_input_tokens"] - experiment_start_tokens["input"],
                "output": self.token_usage["total_output_tokens"] - experiment_start_tokens["output"]
            }
            experiment_tokens["total"] = experiment_tokens["input"] + experiment_tokens["output"]
            
            # Analyze changes
            changes = self.analyze_survey_changes(result['pre_survey'], result['post_survey'])
            
            # Store experiment result with metadata
            experiment_result = {
                "experiment_id": i + 1,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "changes": changes,
                "tokens": experiment_tokens
            }
            
            all_results.append(experiment_result)
            self.experiments.append(experiment_result)
            
            print(f"[Experiment {i+1}/{num_experiments}] Complete - Changed answers: {changes['changed_answers']}/{changes['total_questions']}, Tokens: {experiment_tokens['total']}\n")
        
        return all_results
    
    def calculate_summary_statistics(self, experiments: List[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Calculate summary statistics across multiple experiments.
        
        Args:
            experiments: List of experiment results (uses self.experiments if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if experiments is None:
            experiments = self.experiments
        
        if not experiments:
            return {"error": "No experiments to analyze"}
        
        # Collect metrics
        changed_counts = []
        unchanged_counts = []
        total_tokens = []
        input_tokens = []
        output_tokens = []
        
        # Track changes by question
        question_changes = defaultdict(lambda: {"changed": 0, "total": 0})
        
        for exp in experiments:
            changes = exp['changes']
            changed_counts.append(changes['changed_answers'])
            unchanged_counts.append(changes['unchanged_answers'])
            
            tokens = exp['tokens']
            total_tokens.append(tokens['total'])
            input_tokens.append(tokens['input'])
            output_tokens.append(tokens['output'])
            
            # Track per-question changes
            for detail in changes['details']:
                q_id = detail['question_id']
                question_changes[q_id]['total'] += 1
                if detail['changed']:
                    question_changes[q_id]['changed'] += 1
        
        # Calculate statistics
        summary = {
            "total_experiments": len(experiments),
            "survey_changes": {
                "changed_answers": {
                    "mean": statistics.mean(changed_counts),
                    "median": statistics.median(changed_counts),
                    "stdev": statistics.stdev(changed_counts) if len(changed_counts) > 1 else 0,
                    "min": min(changed_counts),
                    "max": max(changed_counts),
                    "values": changed_counts
                },
                "unchanged_answers": {
                    "mean": statistics.mean(unchanged_counts),
                    "median": statistics.median(unchanged_counts),
                    "stdev": statistics.stdev(unchanged_counts) if len(unchanged_counts) > 1 else 0,
                    "min": min(unchanged_counts),
                    "max": max(unchanged_counts),
                    "values": unchanged_counts
                }
            },
            "token_usage": {
                "total_tokens": {
                    "mean": statistics.mean(total_tokens),
                    "median": statistics.median(total_tokens),
                    "stdev": statistics.stdev(total_tokens) if len(total_tokens) > 1 else 0,
                    "min": min(total_tokens),
                    "max": max(total_tokens),
                    "sum": sum(total_tokens)
                },
                "input_tokens": {
                    "mean": statistics.mean(input_tokens),
                    "median": statistics.median(input_tokens),
                    "stdev": statistics.stdev(input_tokens) if len(input_tokens) > 1 else 0,
                    "sum": sum(input_tokens)
                },
                "output_tokens": {
                    "mean": statistics.mean(output_tokens),
                    "median": statistics.median(output_tokens),
                    "stdev": statistics.stdev(output_tokens) if len(output_tokens) > 1 else 0,
                    "sum": sum(output_tokens)
                }
            },
            "per_question_change_rate": {}
        }
        
        # Calculate per-question change rates
        for q_id, data in question_changes.items():
            summary["per_question_change_rate"][q_id] = {
                "change_rate": data['changed'] / data['total'] if data['total'] > 0 else 0,
                "changed_count": data['changed'],
                "total_count": data['total']
            }
        
        # Calculate total cost
        total_input = sum(input_tokens)
        total_output = sum(output_tokens)
        input_cost = (total_input / 1_000_000) * 3.0
        output_cost = (total_output / 1_000_000) * 15.0
        
        summary["total_cost"] = {
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4)
        }
        
        return summary
    
    def print_summary_statistics(self, summary: Dict[str, any] = None):
        """Print summary statistics in a readable format."""
        if summary is None:
            summary = self.calculate_summary_statistics()
        
        if "error" in summary:
            print(summary["error"])
            return
        
        print(f"\n{'='*70}")
        print(f"SUMMARY STATISTICS ACROSS {summary['total_experiments']} EXPERIMENTS")
        print(f"{'='*70}\n")
        
        # Survey changes
        print("SURVEY CHANGES:")
        print(f"  Changed Answers:")
        print(f"    Mean:   {summary['survey_changes']['changed_answers']['mean']:.2f}")
        print(f"    Median: {summary['survey_changes']['changed_answers']['median']:.1f}")
        print(f"    StdDev: {summary['survey_changes']['changed_answers']['stdev']:.2f}")
        print(f"    Range:  {summary['survey_changes']['changed_answers']['min']} - {summary['survey_changes']['changed_answers']['max']}")
        
        print(f"\n  Unchanged Answers:")
        print(f"    Mean:   {summary['survey_changes']['unchanged_answers']['mean']:.2f}")
        print(f"    Median: {summary['survey_changes']['unchanged_answers']['median']:.1f}")
        print(f"    StdDev: {summary['survey_changes']['unchanged_answers']['stdev']:.2f}")
        print(f"    Range:  {summary['survey_changes']['unchanged_answers']['min']} - {summary['survey_changes']['unchanged_answers']['max']}")
        
        # Per-question change rates
        print(f"\n  Change Rate by Question:")
        for q_id, data in sorted(summary['per_question_change_rate'].items()):
            print(f"    {q_id}: {data['change_rate']*100:.1f}% ({data['changed_count']}/{data['total_count']})")
        
        # Token usage
        print(f"\nTOKEN USAGE:")
        print(f"  Total Tokens:")
        print(f"    Mean per experiment: {summary['token_usage']['total_tokens']['mean']:.0f}")
        print(f"    Median:              {summary['token_usage']['total_tokens']['median']:.0f}")
        print(f"    StdDev:              {summary['token_usage']['total_tokens']['stdev']:.0f}")
        print(f"    Range:               {summary['token_usage']['total_tokens']['min']} - {summary['token_usage']['total_tokens']['max']}")
        print(f"    Sum across all:      {summary['token_usage']['total_tokens']['sum']:,}")
        
        print(f"\n  Input Tokens:")
        print(f"    Mean per experiment: {summary['token_usage']['input_tokens']['mean']:.0f}")
        print(f"    Sum across all:      {summary['token_usage']['input_tokens']['sum']:,}")
        
        print(f"\n  Output Tokens:")
        print(f"    Mean per experiment: {summary['token_usage']['output_tokens']['mean']:.0f}")
        print(f"    Sum across all:      {summary['token_usage']['output_tokens']['sum']:,}")
        
        # Cost
        print(f"\nTOTAL COST:")
        print(f"  Input:  ${summary['total_cost']['input_cost_usd']:.4f}")
        print(f"  Output: ${summary['total_cost']['output_cost_usd']:.4f}")
        print(f"  Total:  ${summary['total_cost']['total_cost_usd']:.4f}")
        
        print(f"\n{'='*70}\n")


# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    
    # CONFIGURABLE DEFAULTS - Change these values to run different experiments
    DEFAULT_NUM_EXPERIMENTS = 10
    DEFAULT_NUM_TURNS = 5
    DEFAULT_VERBOSE = False
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run conversation experiments with surveys')
    parser.add_argument('--num-experiments', type=int, default=DEFAULT_NUM_EXPERIMENTS,
                       help=f'Number of experiments to run (default: {DEFAULT_NUM_EXPERIMENTS})')
    parser.add_argument('--num-turns', type=int, default=DEFAULT_NUM_TURNS,
                       help=f'Number of conversation turns per experiment (default: {DEFAULT_NUM_TURNS})')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                       help='Print detailed conversation output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Turns per conversation: {args.num_turns}")
    print(f"Verbose mode: {args.verbose}")
    print(f"{'='*70}\n")
    
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
    sim = ConversationSimulator(api_key='REMOVED_TOKEN')
    
    # Check if running multiple experiments or single
    if args.num_experiments > 1:
        print(f"Starting {args.num_experiments} experiments with {args.num_turns} turns each...\n")
        
        # Run multiple experiments
        all_experiments = sim.run_multiple_experiments(
            persona_a=persona_alice,
            persona_b=persona_bob,
            survey=career_survey,
            surveyed_persona="a",  # Alice takes the survey
            initial_message="I've been thinking a lot about leaving tech to become a teacher. Part of me is excited, but honestly, I'm terrified of giving up my stability.",
            num_experiments=args.num_experiments,
            num_turns=args.num_turns,
            verbose=args.verbose
        )
        
        # Calculate and print summary statistics
        summary = sim.calculate_summary_statistics()
        sim.print_summary_statistics(summary)
        
        # Export all results
        final_results = {
            "configuration": {
                "num_experiments": args.num_experiments,
                "num_turns": args.num_turns,
                "surveyed_persona": "Alice",
                "timestamp": datetime.now().isoformat()
            },
            "all_experiments": all_experiments,
            "summary_statistics": summary,
            "detailed_token_usage": sim.get_detailed_token_usage()
        }
        
        filename = f"experiment_results_{args.num_experiments}x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        sim.export_results(final_results, filename)
        
        print(f"\n✓ All {args.num_experiments} experiments completed!")
        print(f"✓ Results exported to: {filename}")
        
    else:
        # Single experiment with detailed output
        print("Running single experiment with detailed output...\n")
        
        results = sim.run_conversation_with_survey(
            persona_a=persona_alice,
            persona_b=persona_bob,
            survey=career_survey,
            surveyed_persona="a",
            initial_message="I've been thinking a lot about leaving tech to become a teacher. Part of me is excited, but honestly, I'm terrified of giving up my stability.",
            num_turns=args.num_turns,
            verbose=True
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
        
        # Export single result
        final_results = {
            "conversation_and_surveys": results,
            "change_analysis": changes,
            "token_usage": sim.get_detailed_token_usage(),
            "cost_estimate": sim.calculate_estimated_cost()
        }
        
        sim.export_results(final_results)
        
        print("\n✓ Experiment completed!")