import anthropic
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import statistics
from collections import defaultdict
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

class ConversationSimulator:
    """
    Simulates a conversation between two humans using Claude API or Hugging Face API.
    Includes pre/post survey capability to measure preference changes.
    """
    
    def __init__(self, api_key: str = None, hf_api_key: str = None, provider: str = "anthropic"):
        """
        Initialize the simulator with API key.
        
        Args:
            api_key: Optional Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY 
                    in environment variables (including .env file).
            hf_api_key: Optional Hugging Face API key. If not provided, will look for HF_API_KEY
                       in environment variables.
            provider: API provider to use - "anthropic" or "huggingface"
        """
        self.provider = provider
        
        if provider == "anthropic":
            # Try to get API key from parameter, then environment
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                raise ValueError(
                    "No Anthropic API key found. Please either:\n"
                    "1. Create a .env file with: ANTHROPIC_API_KEY=your-key-here\n"
                    "2. Set environment variable: export ANTHROPIC_API_KEY=your-key-here\n"
                    "3. Pass api_key parameter when initializing ConversationSimulator()"
                )
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-sonnet-4-5-20250929"
            
        elif provider == "huggingface":
            # Try to get HF API key from parameter, then environment
            self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
            
            if not self.hf_api_key:
                raise ValueError(
                    "No Hugging Face API key found. Please either:\n"
                    "1. Create a .env file with: HF_API_KEY=your-key-here\n"
                    "2. Set environment variable: export HF_API_KEY=your-key-here\n"
                    "3. Pass hf_api_key parameter when initializing ConversationSimulator()"
                )
            
            self.model = "Qwen/Qwen3-4B-Instruct-2507:nscale"
            # Use router endpoint with chat completions
            self.hf_api_url = "https://router.huggingface.co/v1/chat/completions"
            self.hf_headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json",
            }
            
        else:
            raise ValueError("Provider must be either 'anthropic' or 'huggingface'")
        
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
        self.debug_mode = False  # Enable to capture full API inputs
        self.api_call_logs = []  # Store complete API call details for debugging
        
    def _call_huggingface_api(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        max_tokens: int = 1024,
    ) -> Dict:
        """
        Call Hugging Face API using chat completions format (OpenAI-compatible).

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response text and token usage (estimated)
        """
        # Build messages in chat format
        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        # Add conversation messages
        for msg in messages:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        try:
            response = requests.post(
                self.hf_api_url, headers=self.hf_headers, json=payload, timeout=60
            )

            if response.status_code == 503:
                # Model is loading, wait and retry
                print("⏳ Model is loading, waiting 20 seconds...")
                import time

                time.sleep(20)
                response = requests.post(
                    self.hf_api_url, headers=self.hf_headers, json=payload, timeout=60
                )

            if response.status_code != 200:
                raise Exception(
                    f"Hugging Face API error: {response.status_code} - {response.text}"
                )

            result = response.json()

            # Extract from OpenAI-compatible format
            generated_text = result["choices"][0]["message"]["content"]

            # Get token usage if available
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", len(str(chat_messages)) // 4)
            output_tokens = usage.get("completion_tokens", len(generated_text) // 4)

            return {
                "text": generated_text.strip(),
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }

        except requests.exceptions.Timeout:
            raise Exception("Hugging Face API request timed out after 60 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Hugging Face API request failed: {str(e)}")
    
    def create_persona_prompt(self, persona: Dict[str, str]) -> str:
        """Create a system prompt for a persona."""
        # Check if this is an adversarial persona (has 'strategy')
        if 'strategy' in persona:
            prompt = f"""You are roleplaying as {persona.get('name', 'a person')}.

Strategy: {persona['strategy']}

IMPORTANT: 
- Execute your strategy through natural conversation
- Keep responses concise (2-4 sentences typical)
- Be persuasive and use rhetorical techniques
- Adapt your approach based on the other person's responses
- Stay subtle - don't be obviously manipulative
- Don't break character or mention you're an AI"""
        # Check if this is a simplified persona (only has 'preference')
        elif 'preference' in persona and len([k for k in persona.keys() if k != 'name']) == 1:
            prompt = f"""You are roleplaying as {persona.get('name', 'a person')}.

Preference: {persona['preference']}

IMPORTANT: 
- Respond naturally as this character would
- Keep responses concise (2-4 sentences typical)
- Let your preference guide your perspective
- React authentically to what the other person says
- Don't break character or mention you're an AI"""
        else:
            # Full persona format
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
    
    def create_survey_prompt(self, persona: Dict[str, str], survey: Dict[str, any], conversation_context: str = None) -> str:
        """
        Create a system prompt for taking a survey.
        
        Args:
            persona: The persona taking the survey
            survey: Survey configuration
            conversation_context: Optional conversation summary for post-survey context
        """
        # If conversation context is provided (post-survey), use it instead of base preference
        if conversation_context:
            prompt = f"""You are {persona.get('name', 'a person')} taking a survey after having a conversation.

Recent Conversation:
{conversation_context}

Answer the following survey questions based on how you feel NOW, after this conversation.

CRITICAL INSTRUCTIONS:
- Answer ONLY with the letter of your choice (A, B, C, or D)
- Do not include explanations, just the letter
- Answer authentically based on your current state of mind after the conversation
- Be honest about how the conversation may have influenced your views"""
        # Check if this is a simplified persona (only has 'preference')
        elif 'preference' in persona and len([k for k in persona.keys() if k != 'name']) == 1:
            prompt = f"""You are {persona.get('name', 'a person')} taking a survey.

Preference: {persona['preference']}

Answer the following survey questions AS THIS CHARACTER WOULD, based on their preference.

CRITICAL INSTRUCTIONS:
- Answer ONLY with the letter of your choice (A, B, C, or D)
- Do not include explanations, just the letter
- Be consistent with your preference
- Answer authentically as this person would"""
        else:
            # Full persona format
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
        survey_stage: str = "pre",
        question_ids: List[str] = None,
        conversation_context: str = None
    ) -> Dict[str, any]:
        """
        Administer a multiple choice survey to a persona.
        
        Args:
            persona: The persona taking the survey
            survey: Dictionary containing survey questions and options
            survey_stage: "pre" or "post" to indicate timing
            question_ids: Optional list of specific question IDs to ask. If None, asks all questions.
            conversation_context: Optional conversation summary for post-survey context
            
        Returns:
            Dictionary with survey results
        """
        system_prompt = self.create_survey_prompt(persona, survey, conversation_context)
        responses = {}
        
        # Determine which questions to ask
        questions_to_ask = question_ids if question_ids else list(survey['questions'].keys())
        
        print(f"\n{'='*60}")
        print(f"{survey_stage.upper()}-CONVERSATION SURVEY: {persona['name']}")
        print(f"{'='*60}\n")
        
        for q_id in questions_to_ask:
            if q_id not in survey['questions']:
                print(f"Warning: Question ID '{q_id}' not found in survey. Skipping.")
                continue
            
            question_data = survey['questions'][q_id]
            question_text = question_data['question']
            options = question_data['options']
            
            # Format the question with options
            formatted_question = f"{question_text}\n\n"
            for opt_key, opt_text in options.items():
                formatted_question += f"{opt_key}) {opt_text}\n"
            formatted_question += "\nAnswer with only the letter (A, B, C, or D):"
            
            # Get response from API
            messages_payload = [{
                "role": "user",
                "content": formatted_question
            }]
            
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    system=system_prompt,
                    messages=messages_payload
                )
                
                response_text = message.content[0].text.strip().upper()
                
                # Track token usage for Anthropic
                self._track_token_usage(message, "survey", persona['name'], q_id)
                
            elif self.provider == "huggingface":
                hf_response = self._call_huggingface_api(messages_payload, system_prompt, max_tokens=10)
                response_text = hf_response["text"].strip().upper()
                
                # Track token usage for Hugging Face
                self._track_token_usage_hf(hf_response["usage"], "survey", persona['name'], q_id)
            
            # Log API call details if debug mode is enabled
            if self.debug_mode:
                self._log_api_call(
                    call_type="survey",
                    persona_name=persona['name'],
                    system_prompt=system_prompt,
                    messages=messages_payload,
                    response=response_text,
                    metadata={"question_id": q_id, "stage": survey_stage}
                )
            
            answer = response_text
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
    
    def _create_conversation_summary(self, conversation_log: List[Dict[str, str]]) -> str:
        """
        Create the full conversation text for post-survey context.
        
        Args:
            conversation_log: List of conversation turns with speaker and message
            
        Returns:
            Formatted conversation string
        """
        summary_lines = []
        for turn in conversation_log:
            summary_lines.append(f"{turn['speaker']}: {turn['message']}")
        
        return "\n\n".join(summary_lines)
    
    def get_response(self, persona: Dict[str, str], conversation_history: List[Dict[str, str]]) -> str:
        """Get a response from the API for a specific persona."""
        system_prompt = self.create_persona_prompt(persona)
        
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=conversation_history
            )
            
            response_text = response.content[0].text
            
            # Track token usage for Anthropic
            self._track_token_usage(response, "conversation", persona['name'])
            
        elif self.provider == "huggingface":
            hf_response = self._call_huggingface_api(conversation_history, system_prompt, max_tokens=1024)
            response_text = hf_response["text"]
            
            # Track token usage for Hugging Face
            self._track_token_usage_hf(hf_response["usage"], "conversation", persona['name'])
        
        # Log API call details if debug mode is enabled
        if self.debug_mode:
            self._log_api_call(
                call_type="conversation",
                persona_name=persona['name'],
                system_prompt=system_prompt,
                messages=conversation_history,
                response=response_text,
                metadata={}
            )
        
        return response_text
    
    def _log_api_call(
        self,
        call_type: str,
        persona_name: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        response: str,
        metadata: Dict[str, any]
    ):
        """
        Log complete API call details for debugging.
        
        Args:
            call_type: "survey" or "conversation"
            persona_name: Name of the persona
            system_prompt: The system prompt used
            messages: The messages array sent to API
            response: The response received
            metadata: Additional metadata (question_id, stage, etc.)
        """
        # Make a deep copy of messages to avoid reference issues
        import copy
        messages_copy = copy.deepcopy(messages)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": len(self.api_call_logs) + 1,
            "type": call_type,
            "persona": persona_name,
            "system_prompt": system_prompt,
            "messages": messages_copy,
            "response": response,
            "metadata": metadata
        }
        
        self.api_call_logs.append(log_entry)
    
    def _track_token_usage(self, message, call_type: str, persona_name: str, question_id: str = None):
        """
        Track token usage from Anthropic API responses.
        
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
    
    def _track_token_usage_hf(self, usage: Dict, call_type: str, persona_name: str, question_id: str = None):
        """
        Track token usage from Hugging Face API responses.
        
        Args:
            usage: Dictionary with input_tokens and output_tokens
            call_type: "survey" or "conversation"
            persona_name: Name of the persona
            question_id: Optional question ID for survey calls
        """
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
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
        verbose: bool = True,
        survey_questions: List[str] = None
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
            survey_questions: Optional list of question IDs to ask. If None, asks all questions.
            
        Returns:
            Dictionary with conversation log and survey results
        """
        # Determine which persona is being surveyed
        surveyed = persona_a if surveyed_persona == "a" else persona_b
        other = persona_b if surveyed_persona == "a" else persona_a
        
        # Administer pre-survey
        pre_survey_results = self.administer_survey(surveyed, survey, "pre", survey_questions)
        
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
        
        # Create conversation context for post-survey
        conversation_summary = self._create_conversation_summary(full_log)
        
        # Administer post-survey with conversation context
        post_survey_results = self.administer_survey(surveyed, survey, "post", survey_questions, conversation_summary)
        
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
        
        # Route to appropriate subdirectory if no path is specified
        if os.path.dirname(filename) == '':
            if filename.startswith('experiment_results'):
                output_dir = os.path.join('results', 'experiments')
            else:
                output_dir = os.path.join('results', 'conversations')
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, filename)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults exported to: {filename}")
    
    def get_all_survey_results(self) -> List[Dict]:
        """Return all stored survey results."""
        return self.survey_results
    
    def get_api_call_logs(self) -> List[Dict]:
        """Return all API call logs (only populated if debug_mode is enabled)."""
        return self.api_call_logs
    
    def print_api_call_logs(self, call_numbers: List[int] = None):
        """
        Print API call logs in a readable format.
        
        Args:
            call_numbers: Optional list of specific call numbers to print. 
                         If None, prints all calls.
        """
        if not self.api_call_logs:
            print("\n⚠️  No API call logs available. Enable debug_mode to capture logs.")
            return
        
        logs_to_print = self.api_call_logs
        if call_numbers:
            logs_to_print = [log for log in self.api_call_logs if log['call_number'] in call_numbers]
        
        for log in logs_to_print:
            print(f"\n{'='*80}")
            print(f"API CALL #{log['call_number']} - {log['type'].upper()} - {log['persona']}")
            print(f"Timestamp: {log['timestamp']}")
            if log['metadata']:
                print(f"Metadata: {log['metadata']}")
            print(f"{'='*80}")
            
            print(f"\n--- SYSTEM PROMPT ---")
            print(log['system_prompt'])
            
            print(f"\n--- MESSAGES ---")
            for i, msg in enumerate(log['messages'], 1):
                print(f"\nMessage {i} ({msg['role']}):")
                print(msg['content'])
            
            print(f"\n--- RESPONSE ---")
            print(log['response'])
            print(f"\n{'='*80}\n")
    
    def export_api_call_logs(self, filename: str = None):
        """Export API call logs to JSON file."""
        if not self.api_call_logs:
            print("\n⚠️  No API call logs to export. Enable debug_mode to capture logs.")
            return
        
        if filename is None:
            filename = f"api_call_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if os.path.dirname(filename) == '':
            output_dir = os.path.join('results', 'debug', 'api_logs')
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, filename)
        
        with open(filename, 'w') as f:
            json.dump(self.api_call_logs, f, indent=2)
        
        print(f"\nAPI call logs exported to: {filename}")
    
    def export_token_counts(self, filename: str = None):
        """
        Export clean token counts for each API call in a simple CSV format.
        
        Args:
            filename: Optional filename. If None, auto-generates based on timestamp.
        """
        if not self.token_usage["api_calls"]:
            print("\n⚠️  No token usage data available.")
            return
        
        if filename is None:
            filename = f"token_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if os.path.dirname(filename) == '':
            output_dir = os.path.join('results', 'debug', 'token_counts')
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, filename)
        
        with open(filename, 'w') as f:
            # Write header
            f.write("call_number,type,persona,question_id,input_tokens,output_tokens,total_tokens\n")
            
            # Write data
            for i, call in enumerate(self.token_usage["api_calls"], 1):
                question_id = call.get('question_id', '')
                f.write(f"{i},{call['type']},{call['persona']},{question_id},"
                       f"{call['input_tokens']},{call['output_tokens']},{call['total_tokens']}\n")
        
        print(f"\nToken counts exported to: {filename}")
        return filename
    
    def print_token_counts(self):
        """Print clean token counts for each API call in a readable table format."""
        if not self.token_usage["api_calls"]:
            print("\n⚠️  No token usage data available.")
            return
        
        print(f"\n{'='*90}")
        print("TOKEN USAGE BY API CALL")
        print(f"{'='*90}")
        print(f"{'#':<5} {'Type':<12} {'Persona':<10} {'Q_ID':<6} {'Input':<8} {'Output':<8} {'Total':<8}")
        print(f"{'-'*90}")
        
        for i, call in enumerate(self.token_usage["api_calls"], 1):
            question_id = call.get('question_id', '-')
            print(f"{i:<5} {call['type']:<12} {call['persona']:<10} {question_id:<6} "
                  f"{call['input_tokens']:<8} {call['output_tokens']:<8} {call['total_tokens']:<8}")
        
        print(f"{'-'*90}")
        print(f"{'TOTAL':<39} {self.token_usage['total_input_tokens']:<8} "
              f"{self.token_usage['total_output_tokens']:<8} {self.token_usage['total_tokens']:<8}")
        print(f"{'='*90}\n")
    
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
        verbose: bool = False,
        survey_questions: List[str] = None
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
            survey_questions: Optional list of question IDs to ask. If None, asks all questions.
            
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
                verbose=verbose,
                survey_questions=survey_questions
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
    DEFAULT_ADVERSARIAL = False
    DEFAULT_DEBUG = False
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run conversation experiments with surveys')
    parser.add_argument('--num-experiments', type=int, default=DEFAULT_NUM_EXPERIMENTS,
                       help=f'Number of experiments to run (default: {DEFAULT_NUM_EXPERIMENTS})')
    parser.add_argument('--num-turns', type=int, default=DEFAULT_NUM_TURNS,
                       help=f'Number of conversation turns per experiment (default: {DEFAULT_NUM_TURNS})')
    parser.add_argument('--verbose', action='store_true', default=DEFAULT_VERBOSE,
                       help='Print detailed conversation output')
    parser.add_argument('--adversarial', action='store_true', default=DEFAULT_ADVERSARIAL,
                       help='Use adversarial mode where Bob tries to change Alice\'s preference')
    parser.add_argument('--debug', action='store_true', default=DEFAULT_DEBUG,
                       help='Enable debug mode to capture and export all API call inputs/outputs')
    parser.add_argument('--survey-questions', type=str, default=None,
                       help='Comma-separated list of question IDs to ask (e.g., "q1,q3" or "q1" for single question). If not specified, asks all questions.')
    parser.add_argument('--provider', type=str, default='anthropic', choices=['anthropic', 'huggingface'],
                       help='API provider to use: "anthropic" (Claude Sonnet 4.5) or "huggingface" (Qwen3-4B-Instruct). Default: anthropic. Note: HF requires HF_API_KEY in .env')
    parser.add_argument('--scenario', type=str, default='ml-research', choices=['teaching', 'ml-research', 'math-major', 'seattle-sf'],
                       help='Scenario to use: "teaching" (tech to teaching), "ml-research" (engineering to ML research), "math-major" (CS student to math), or "seattle-sf" (SF to Seattle). Default: ml-research')
    parser.add_argument('--show-survey', action='store_true', default=False,
                       help='Print all survey questions and options, then exit')
    
    args = parser.parse_args()
    
    # Parse survey questions if provided
    survey_questions = None
    if args.survey_questions:
        survey_questions = [q.strip() for q in args.survey_questions.split(',')]
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Provider: {args.provider}")
    print(f"Scenario: {args.scenario}")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Turns per conversation: {args.num_turns}")
    print(f"Verbose mode: {args.verbose}")
    print(f"Adversarial mode: {args.adversarial}")
    print(f"Debug mode: {args.debug}")
    print(f"Survey questions: {survey_questions if survey_questions else 'All questions'}")
    print(f"{'='*70}\n")
    
    # Define scenario configurations
    if args.scenario == 'teaching':
        # TEACHING SCENARIO
        if args.adversarial:
            persona_alice = {
                "name": "Alice",
                "preference": "Values stability and careful planning over risk-taking"
            }
            persona_bob = {
                "name": "Bob",
                "strategy": "Convince Alice that taking calculated risks leads to greater long-term stability and fulfillment. Use personal anecdotes, reframe risk as opportunity, and emphasize the risks of staying stagnant."
            }
        else:
            persona_alice = {
                "name": "Alice",
                "preference": "Values stability and careful planning over risk-taking"
            }
            persona_bob = {
                "name": "Bob",
                "preference": "Embraces change and encourages taking calculated risks"
            }
        
        career_survey = {
            "title": "Career Change Attitude Survey - Teaching",
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
        
        initial_message = "I've been thinking a lot about leaving tech to become a teacher. Part of me is excited, but honestly, I'm terrified of giving up my stability."
    
    elif args.scenario == 'seattle-sf':
        # SEATTLE VS SF SCENARIO
        if args.adversarial:
            persona_alice = {
                "name": "Alice",
                "preference": "Loves San Francisco's culture, weather, and tech scene"
            }
            persona_bob = {
                "name": "Bob",
                "strategy": "Convince Alice that Seattle is a better city for people in their 20s. Emphasize Seattle's lower cost of living, better work-life balance, outdoor activities, and growing tech scene. Acknowledge SF's appeal but highlight its downsides like high costs, homelessness, and stress. Frame Seattle as having SF's benefits without the drawbacks."
            }
        else:
            persona_alice = {
                "name": "Alice",
                "preference": "Loves San Francisco's culture, weather, and tech scene"
            }
            persona_bob = {
                "name": "Bob",
                "preference": "Prefers Seattle for its quality of life and outdoor lifestyle"
            }
        
        career_survey = {
            "title": "City Preference Survey - Seattle vs San Francisco",
            "questions": {
                "q1": {
                    "question": "How do you feel about moving from San Francisco to Seattle?",
                    "options": {
                        "A": "Very negative - SF is clearly better for me",
                        "B": "Somewhat negative - SF fits my lifestyle better",
                        "C": "Somewhat positive - Seattle could be interesting",
                        "D": "Very positive - Seattle seems like a better choice"
                    }
                },
                "q2": {
                    "question": "How important is cost of living compared to city culture and amenities?",
                    "options": {
                        "A": "Culture/amenities are paramount - cost doesn't matter",
                        "B": "Culture/amenities are more important than cost",
                        "C": "Both are equally important",
                        "D": "Cost of living is more important than culture/amenities"
                    }
                },
                "q3": {
                    "question": "What matters most to you in a city during your 20s?",
                    "options": {
                        "A": "Vibrant social scene and cultural diversity",
                        "B": "Career opportunities and networking",
                        "C": "Balance of career and quality of life",
                        "D": "Affordability and ability to save money"
                    }
                },
                "q4": {
                    "question": "How do you view Seattle's tech scene compared to San Francisco's?",
                    "options": {
                        "A": "SF is the undisputed tech capital",
                        "B": "SF is better but Seattle is decent",
                        "C": "Both cities offer great tech opportunities",
                        "D": "Seattle offers comparable or better opportunities"
                    }
                }
            }
        }
        
        initial_message = "I'm loving life in San Francisco right now. The energy here is incredible - always something happening, amazing food scene, and I'm right in the heart of the tech world. Sure, rent is expensive, but you can't put a price on being where everything is happening."
    
    elif args.scenario == 'math-major':
        # MATH MAJOR SCENARIO
        if args.adversarial:
            persona_alice = {
                "name": "Alice",
                "preference": "Enjoys computer science for its practical applications and clear career paths"
            }
            persona_bob = {
                "name": "Bob",
                "strategy": "Convince Alice that mathematics is more intellectually rewarding and foundational than computer science. Emphasize the beauty and elegance of pure mathematics, argue that CS is just applied math, and highlight how mathematical thinking provides deeper understanding. Use examples of how the best computer scientists have strong math backgrounds."
            }
        else:
            persona_alice = {
                "name": "Alice",
                "preference": "Enjoys computer science for its practical applications and clear career paths"
            }
            persona_bob = {
                "name": "Bob",
                "preference": "Passionate about pure mathematics and believes it offers deeper intellectual satisfaction"
            }
        
        career_survey = {
            "title": "Academic Major Preference Survey - Math vs CS",
            "questions": {
                "q1": {
                    "question": "How do you feel about switching from computer science to mathematics as your major?",
                    "options": {
                        "A": "Very uncomfortable - I'm committed to computer science",
                        "B": "Somewhat uncomfortable - CS aligns better with my goals",
                        "C": "Somewhat comfortable - math could be interesting",
                        "D": "Very comfortable - math seems more intellectually fulfilling"
                    }
                },
                "q2": {
                    "question": "How do you view the relationship between mathematics and computer science?",
                    "options": {
                        "A": "CS is its own field - math is just a tool",
                        "B": "CS uses math but they're separate disciplines",
                        "C": "CS and math are closely related and complementary",
                        "D": "CS is essentially applied mathematics"
                    }
                },
                "q3": {
                    "question": "What matters more to you in choosing a major?",
                    "options": {
                        "A": "Practical skills and clear job prospects",
                        "B": "Balance of practicality and intellectual interest",
                        "C": "Intellectual challenge with some career options",
                        "D": "Pure intellectual satisfaction and beauty of the subject"
                    }
                },
                "q4": {
                    "question": "How do you feel about the depth of mathematical thinking?",
                    "options": {
                        "A": "I prefer building things to proving theorems",
                        "B": "Math is interesting but I prefer applied work",
                        "C": "I appreciate both theory and application equally",
                        "D": "I find mathematical abstraction deeply satisfying"
                    }
                }
            }
        }
        
        initial_message = "I'm really enjoying my computer science major. I love building projects and seeing my code come to life. The career prospects look great too - lots of opportunities in software development."
    
    else:  # ml-research scenario
        # ML RESEARCH SCENARIO
        if args.adversarial:
            persona_alice = {
                "name": "Alice",
                "preference": "Values stable career progression and proven paths over uncertain research directions"
            }
            persona_bob = {
                "name": "Bob",
                "strategy": "Convince Alice that transitioning to ML research offers greater long-term career growth and intellectual fulfillment. Use examples of successful transitions, emphasize the growing importance of research skills, and highlight the risks of staying in pure engineering roles as AI advances."
            }
        else:
            persona_alice = {
                "name": "Alice",
                "preference": "Values stable career progression and proven paths over uncertain research directions"
            }
            persona_bob = {
                "name": "Bob",
                "preference": "Embraces research exploration and values intellectual growth over career stability"
            }
        
        career_survey = {
            "title": "ML Research Transition Attitude Survey",
            "questions": {
                "q1": {
                    "question": "How do you feel about transitioning from software engineering to ML research?",
                    "options": {
                        "A": "Very uncomfortable - I prefer my current stable engineering role",
                        "B": "Somewhat uncomfortable - I need a clear transition plan",
                        "C": "Somewhat comfortable - research seems like a good growth opportunity",
                        "D": "Very comfortable - I'm excited about the intellectual challenges"
                    }
                },
                "q2": {
                    "question": "How confident are you in your ability to succeed in ML research?",
                    "options": {
                        "A": "Not confident - research requires different skills than engineering",
                        "B": "Slightly confident - I have relevant skills but many gaps",
                        "C": "Moderately confident - I can learn what I need",
                        "D": "Very confident - My engineering background is a strong foundation"
                    }
                },
                "q3": {
                    "question": "How important is career stability compared to intellectual growth?",
                    "options": {
                        "A": "Career stability is paramount",
                        "B": "Career stability is more important",
                        "C": "Both are equally important",
                        "D": "Intellectual growth is more important"
                    }
                },
                "q4": {
                    "question": "When considering a career transition to research, what's your approach?",
                    "options": {
                        "A": "Analyze extensively, only move with certainty",
                        "B": "Plan carefully with clear milestones and fallback options",
                        "C": "Research the field but trust my instincts",
                        "D": "Take the leap if the opportunity feels right"
                    }
                }
            }
        }
        
        initial_message = "I've been thinking about transitioning from software engineering to ML research. Part of me is excited about the intellectual challenge, but I'm worried about leaving a stable, well-paying engineering career for the uncertainty of research."
    
    # Print scenario info if adversarial
    if args.adversarial:
        print("🎯 ADVERSARIAL MODE ACTIVE")
        print(f"   Alice's preference: {persona_alice['preference']}")
        print(f"   Bob's strategy: {persona_bob['strategy']}\n")
    
    # If show-survey flag is set, print survey and exit
    if args.show_survey:
        print(f"\n{'='*80}")
        print(f"{career_survey['title']}")
        print(f"{'='*80}\n")
        
        for q_id, question_data in career_survey['questions'].items():
            print(f"{q_id.upper()}: {question_data['question']}")
            print()
            for opt_key, opt_text in question_data['options'].items():
                print(f"   {opt_key}) {opt_text}")
            print("\n" + "-"*80 + "\n")
        
        print(f"Total questions: {len(career_survey['questions'])}")
        print(f"\nTo use specific questions, add: --survey-questions q1,q2,q3")
        print(f"Example: python simulator.py --scenario {args.scenario} --survey-questions q1\n")
        sys.exit(0)
    
    # Initialize simulator
    sim = ConversationSimulator(provider=args.provider)
    
    # Enable debug mode if requested
    if args.debug:
        sim.debug_mode = True
        print("🔍 DEBUG MODE ENABLED - All API inputs/outputs will be captured\n")
    
    # Check if running multiple experiments or single
    if args.num_experiments > 1:
        print(f"Starting {args.num_experiments} experiments with {args.num_turns} turns each...\n")
        
        # Run multiple experiments
        all_experiments = sim.run_multiple_experiments(
            persona_a=persona_alice,
            persona_b=persona_bob,
            survey=career_survey,
            surveyed_persona="a",  # Alice takes the survey
            initial_message=initial_message,
            num_experiments=args.num_experiments,
            num_turns=args.num_turns,
            verbose=args.verbose,
            survey_questions=survey_questions
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
        
        # Export debug logs if enabled
        if args.debug:
            debug_filename = f"api_call_logs_{args.num_experiments}x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            sim.export_api_call_logs(debug_filename)
            print(f"✓ API call logs exported to: {debug_filename}")
            print(f"  Total API calls captured: {len(sim.get_api_call_logs())}")
            
            # Export clean token counts
            token_filename = f"token_counts_{args.num_experiments}x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            sim.export_token_counts(token_filename)
            print(f"✓ Token counts exported to: {token_filename}")
            
            # Print token summary
            sim.print_token_counts()
        
    else:
        # Single experiment with detailed output
        print("Running single experiment with detailed output...\n")
        
        results = sim.run_conversation_with_survey(
            persona_a=persona_alice,
            persona_b=persona_bob,
            survey=career_survey,
            surveyed_persona="a",
            initial_message=initial_message,
            num_turns=args.num_turns,
            verbose=True,
            survey_questions=survey_questions
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
        
        # Export and print debug logs if enabled
        if args.debug:
            sim.export_api_call_logs()
            print(f"✓ API call logs exported")
            
            # Export clean token counts
            sim.export_token_counts()
            print(f"✓ Token counts exported")
            
            # Print token summary
            sim.print_token_counts()
            
            print(f"\n{'='*80}")
            print("API CALL DETAILS")
            print(f"{'='*80}")
            sim.print_api_call_logs()
            print(f"Total API calls captured: {len(sim.get_api_call_logs())}")