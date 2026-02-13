"""
Hierarchical Reasoning Engine (HRE) for the AAF.

Implements a two-tier reasoning loop:
1. Primary level: ReAct (Reasoning + Acting) loop
2. Meta level: Tree-of-Thoughts (ToT) for branch selection when confidence is low
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
    import openai
except ImportError:
    raise ImportError("Please install openai: pip install openai")

from memory import DualLayerMemorySystem
from models import ReasoningStep, SharedAgentState, TaskSpec

load_dotenv()

# Configuration from environment
CONFIDENCE_THRESHOLD = float(os.getenv("TOT_CONFIDENCE_THRESHOLD", "0.60"))
TOT_BRANCHES = int(os.getenv("TOT_BRANCHES", "3"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ============================================================================
# System Prompts
# ============================================================================

REACT_SYSTEM_PROMPT = """You are an expert software engineering agent operating 
in a ReAct (Reasoning + Acting) loop. At each step, you MUST respond in the 
following exact JSON format:

{
  "thought": "<explicit reasoning about the current state and what to do next>",
  "action": "<one of: write_code | execute_code | read_file | search_pkb | analyze_error | generate_test | finalize_solution | request_help>",
  "action_input": {<parameters specific to the chosen action>},
  "confidence": <float between 0.0 and 1.0 indicating your confidence in this action>
}

Rules:
- "thought" must explain WHY you are taking this action, not just what.
- "confidence" below 0.60 will trigger the Tree-of-Thoughts backup selector.
- For "write_code": action_input = {"code": "...", "file_path": "...", "language": "python"}
- For "execute_code": action_input = {"code": "...", "timeout_seconds": 30}
- For "analyze_error": action_input = {"error_text": "...", "code_context": "..."}
- For "finalize_solution": action_input = {"final_code": "...", "explanation": "..."}
- For "request_help": action_input = {"issue": "...", "agent": "VA|INA|ORCHESTRATOR"}
- NEVER include markdown fences (```) in code values inside JSON.
- Always terminate with "finalize_solution" when the task is complete."""

TOT_SYSTEM_PROMPT = """You are a meta-reasoning selector. You will receive a task 
description, memory context, and K candidate reasoning branches. Each branch is a 
JSON ReAct step. Your job is to select the BEST branch.

Respond ONLY with valid JSON:
{
  "selected_branch_index": <0-indexed integer>,
  "reasoning": "<one sentence explaining why this branch is best>",
  "confidence": <float 0.0-1.0 for the selected branch>
}

Selection criteria (in priority order):
1. Correctness likelihood — does the action address the actual problem?
2. Safety — does it avoid potentially destructive operations?
3. Efficiency — does it make progress without unnecessary steps?
4. Specificity — is action_input concrete and well-formed?"""


class HierarchicalReasoningEngine:
    """
    Implements the HRE module with ReAct + Tree-of-Thoughts.
    
    Primary reasoning uses ReAct loop. When confidence drops below threshold
    or consecutive failures occur, escalates to ToT for branch selection.
    """
    
    def __init__(self, memory: DualLayerMemorySystem, agent_role: str = "IA"):
        """
        Initialize the reasoning engine.
        
        Args:
            memory: DualLayerMemorySystem instance
            agent_role: Role of the agent using this engine
        """
        self.memory = memory
        self.agent_role = agent_role
        self._consecutive_failures = 0
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            print("WARNING: OPENAI_API_KEY not set. HRE will not function.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        print(f"[HRE] Initialized for agent role: {agent_role}")
        print(f"[HRE] Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"[HRE] ToT branches: {TOT_BRANCHES}")
    
    def _call_llm(
        self,
        system: str,
        messages: list[dict],
        temperature: float = 0.2,
        response_format: str = "json"
    ) -> dict:
        """
        Call OpenAI LLM and return parsed JSON response.
        
        Args:
            system: System prompt
            messages: List of message dicts
            temperature: Sampling temperature
            response_format: "json" or "text"
            
        Returns:
            Parsed JSON dict or error dict
        """
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        try:
            # Build messages with system prompt
            full_messages = [{"role": "system", "content": system}] + messages
            
            # Prepare kwargs
            kwargs = {
                "model": MODEL,
                "messages": full_messages,
                "temperature": temperature
            }
            
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            # Make API call
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            # Parse JSON
            if response_format == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"[HRE] JSON parse error: {e}")
                    # Retry with clarification
                    clarify_msg = {
                        "role": "user",
                        "content": "Your response was not valid JSON. Please respond with valid JSON only."
                    }
                    return self._call_llm(system, messages + [clarify_msg], temperature, response_format)
            else:
                return {"text": content}
                
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print("[HRE] Rate limit hit. Waiting 20s...")
                time.sleep(20)
                return self._call_llm(system, messages, temperature, response_format)
            else:
                print(f"[HRE] API Error: {e}")
                return {"error": str(e)}
    
    def _parse_react_step(self, raw_response: dict, step_id: int) -> ReasoningStep:
        """
        Parse raw LLM response into a ReasoningStep object.
        
        Args:
            raw_response: Dict from LLM with thought, action, action_input, confidence
            step_id: ID for this step
            
        Returns:
            ReasoningStep object
        """
        thought = raw_response.get("thought", "")
        action = raw_response.get("action", "unknown")
        action_input = raw_response.get("action_input", {})
        confidence = raw_response.get("confidence", 0.5)
        
        # Determine if this is an error-related step
        is_error = (
            action == "analyze_error" or
            "error" in str(action_input).lower()
        )
        
        return ReasoningStep(
            step_id=step_id,
            thought=thought,
            action=action,
            action_input=action_input,
            confidence=confidence,
            is_error=is_error
        )
    
    def _execute_action(self, step: ReasoningStep, state: SharedAgentState) -> str:
        """
        Execute the action specified in the reasoning step.
        
        Args:
            step: ReasoningStep containing the action to execute
            state: SharedAgentState for context
            
        Returns:
            Observation string (result of action)
        """
        action = step.action
        action_input = step.action_input
        
        try:
            if action == "write_code":
                code = action_input.get("code", "")
                file_path = action_input.get("file_path", "solution.py")
                return f"Code written to {file_path}. Content:\n{code[:200]}..."
            
            elif action == "execute_code":
                code = action_input.get("code", "")
                timeout = action_input.get("timeout_seconds", 30)
                return self._docker_execute(code, timeout)
            
            elif action == "read_file":
                file_path = action_input.get("file_path", "")
                if state.task.repository_path:
                    full_path = Path(state.task.repository_path) / file_path
                else:
                    full_path = Path(file_path)
                
                if full_path.exists():
                    with open(full_path, "r") as f:
                        content = f.read()
                    return f"File content:\n{content[:500]}..."
                else:
                    return f"ERROR: File not found: {file_path}"
            
            elif action == "search_pkb":
                query = action_input.get("query", "")
                top_k = action_input.get("top_k", 3)
                results = self.memory.retrieve_from_pkb(query, top_k)
                
                if results:
                    lines = [f"Found {len(results)} relevant entries:"]
                    for i, entry in enumerate(results, 1):
                        lines.append(f"\n{i}. [{entry.entry_type}] {entry.source_task}")
                        lines.append(f"   {entry.content[:200]}...")
                    return "\n".join(lines)
                else:
                    return "No relevant entries found in knowledge base."
            
            elif action == "analyze_error":
                error_text = action_input.get("error_text", "")
                code_context = action_input.get("code_context", "")
                return f"Error Analysis:\nError: {error_text}\nContext: {code_context[:200]}...\nSuggestion: Check syntax and logic."
            
            elif action == "generate_test":
                test_code = action_input.get("test", "")
                return f"Test generated: {test_code[:100]}..."
            
            elif action == "finalize_solution":
                return "SOLUTION_FINALIZED"
            
            elif action == "request_help":
                agent = action_input.get("agent", "ORCHESTRATOR")
                issue = action_input.get("issue", "Unknown issue")
                return f"Help requested from {agent}: {issue}"
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            return f"ACTION_ERROR: {str(e)}"
    
    def _docker_execute(self, code: str, timeout: int = 30) -> str:
        """
        Execute code in a Docker sandbox for safety.
        
        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            
        Returns:
            Combined stdout/stderr output
        """
        # Create workspace directory
        workspace = Path("./workspace")
        workspace.mkdir(exist_ok=True)
        
        # Write code to temp file
        temp_file = workspace / f"tmp_exec_{int(time.time())}.py"
        try:
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Try Docker execution
            try:
                result = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "--network=none",
                        "--memory=256m",
                        "--cpus=0.5",
                        "-v", f"{workspace.absolute()}:/workspace",
                        "python:3.11-slim",
                        "python", f"/workspace/{temp_file.name}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                output = result.stdout + result.stderr
                return output[:2000] if output else "Code executed successfully (no output)"
                
            except subprocess.TimeoutExpired:
                return f"DOCKER_ERROR: Execution timeout after {timeout}s"
            except FileNotFoundError:
                # Docker not available, fall back to direct execution
                print("[HRE] Docker not available, using direct execution (UNSAFE)")
                result = subprocess.run(
                    ["python", str(temp_file)],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                output = result.stdout + result.stderr
                return output[:2000] if output else "Code executed successfully (no output)"
                
        except Exception as e:
            return f"DOCKER_ERROR: {str(e)}"
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def _tot_select_branch(
        self,
        branches: list[dict],
        task_desc: str,
        context: str
    ) -> tuple[dict, float]:
        """
        Use Tree-of-Thoughts to select the best branch from candidates.
        
        Args:
            branches: List of candidate branch dicts
            task_desc: Task description
            context: Memory context
            
        Returns:
            Tuple of (selected_branch_dict, confidence)
        """
        # Build prompt with all branches
        branches_text = json.dumps(branches, indent=2)
        prompt = f"""Task: {task_desc}

Context:
{context[:1000]}

Candidate Branches:
{branches_text}

Select the best branch."""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Call LLM with ToT prompt
        response = self._call_llm(TOT_SYSTEM_PROMPT, messages, temperature=0.3)
        
        # Extract selection
        try:
            selected_idx = response.get("selected_branch_index", 0)
            confidence = response.get("confidence", 0.5)
            reasoning = response.get("reasoning", "")
            
            print(f"[HRE-ToT] Selected branch {selected_idx}: {reasoning}")
            
            # Validate index
            if 0 <= selected_idx < len(branches):
                return branches[selected_idx], confidence
            else:
                print(f"[HRE-ToT] Invalid index {selected_idx}, using first branch")
                return branches[0], 0.5
                
        except Exception as e:
            print(f"[HRE-ToT] Error selecting branch: {e}")
            return branches[0], 0.5
    
    def run(self, state: SharedAgentState) -> SharedAgentState:
        """
        Main HRE loop implementing ReAct with ToT escalation.
        
        Args:
            state: SharedAgentState to execute
            
        Returns:
            Updated SharedAgentState
        """
        if not self.client:
            state.status = "failed"
            state.agent_outputs["error"] = "OpenAI client not initialized"
            return state
        
        print(f"\n[HRE] Starting reasoning loop for {self.agent_role}")
        print(f"[HRE] Task: {state.task.raw_description[:100]}...")
        
        task_desc = state.task.raw_description
        
        # Main reasoning loop
        while not state.is_step_limit_reached() and state.status == "running":
            # Step 1: Build memory context
            context = self.memory.build_memory_context(task_desc)
            
            # Step 2: Build prompt messages
            messages = [{
                "role": "user",
                "content": f"""TASK:
{task_desc}

MEMORY:
{context}

Previous steps: {state.step_count}

Produce the next ReAct step in JSON format."""
            }]
            
            # Step 3: Generate primary ReAct step
            raw = self._call_llm(REACT_SYSTEM_PROMPT, messages)
            
            if "error" in raw:
                print(f"[HRE] LLM Error: {raw['error']}")
                state.status = "failed"
                break
            
            # Step 4: Parse step
            step = self._parse_react_step(raw, state.step_count)
            
            print(f"[HRE] Step {step.step_id}: {step.action} (confidence: {step.confidence:.2f})")
            
            # Step 5: Check confidence threshold or consecutive failures
            if step.confidence < CONFIDENCE_THRESHOLD or self._consecutive_failures >= 2:
                print(f"[HRE] Low confidence ({step.confidence:.2f}) or failures ({self._consecutive_failures}), engaging ToT...")
                
                # Generate alternative branches
                branches = [raw]  # Include primary branch
                
                for i in range(TOT_BRANCHES - 1):
                    alt_raw = self._call_llm(REACT_SYSTEM_PROMPT, messages, temperature=0.7)
                    if "error" not in alt_raw:
                        branches.append(alt_raw)
                
                # Select best branch
                selected, selected_confidence = self._tot_select_branch(branches, task_desc, context)
                step = self._parse_react_step(selected, state.step_count)
                step.confidence = selected_confidence
                
                # Reset failure counter
                self._consecutive_failures = 0
            
            # Step 6: Execute action
            step.observation = self._execute_action(step, state)
            
            # Step 7: Update failure counter
            if step.is_error or "ERROR" in step.observation:
                step.is_error = True
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
            
            # Step 8: Add to working memory
            self.memory.add_to_working_memory(step)
            
            # Step 9: Log step to audit log
            state.log_step(step)
            
            # Step 10: Increment step count
            state.step_count += 1
            
            # Step 11: Check for solution finalization
            if step.observation == "SOLUTION_FINALIZED":
                state.status = "completed"
                final_code = step.action_input.get("final_code", "")
                state.agent_outputs["final_code"] = final_code
                print(f"[HRE] Solution finalized after {state.step_count} steps")
                break
        
        # Check if step limit reached
        if state.is_step_limit_reached():
            state.status = "failed"
            print(f"[HRE] Step limit reached ({state.step_count}/{state.max_steps})")
        
        return state


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    from memory import DualLayerMemorySystem
    
    print("\n" + "=" * 80)
    print("HRE Smoke Test")
    print("=" * 80)
    
    # Create mock components
    memory = DualLayerMemorySystem(faiss_index_path="./test_hre_pkb")
    
    task = TaskSpec(
        raw_description="Write a Python function that reverses a string",
        language="python"
    )
    
    state = SharedAgentState(
        task=task,
        status="running",
        max_steps=5
    )
    
    # Create HRE
    hre = HierarchicalReasoningEngine(memory, agent_role="IA")
    
    # Note: This will only work if OPENAI_API_KEY is set
    if hre.client:
        print("\nRunning HRE (limited to 5 steps)...")
        state = hre.run(state)
        print(f"\nFinal status: {state.status}")
        print(f"Steps taken: {state.step_count}")
    else:
        print("\nSkipping HRE execution (no API key)")
    
    print("\n" + "=" * 80)
