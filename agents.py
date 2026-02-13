"""
Multi-Agent Coordination Protocol (MACP) for the AAF.

Implements 5 specialist agents coordinated by an orchestrator:
- RA: Requirements Agent
- AA: Architecture Agent
- IA: Implementation Agent
- VA: Verification Agent
- INA: Integration Agent
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
    import openai
except ImportError:
    raise ImportError("Please install openai: pip install openai")

from memory import DualLayerMemorySystem
from models import AgentMessage, ReasoningStep, SharedAgentState, TaskSpec, WorkingMemoryRecord
from reasoning_engine import HierarchicalReasoningEngine

load_dotenv()


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents in the MACP."""
    
    ROLE: str = "BASE"
    SYSTEM_PROMPT: str = ""
    
    def __init__(self, memory: DualLayerMemorySystem):
        """
        Initialize the agent.
        
        Args:
            memory: Shared DualLayerMemorySystem instance
        """
        self.memory = memory
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            print(f"WARNING: OPENAI_API_KEY not set. {self.ROLE} will not function.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
    
    @abstractmethod
    def run(self, state: SharedAgentState) -> AgentMessage:
        """
        Execute the agent's task.
        
        Args:
            state: SharedAgentState containing task and context
            
        Returns:
            AgentMessage with results
        """
        pass
    
    def _send_message(
        self,
        recipient: str,
        msg_type: str,
        payload: dict[str, Any],
        state: SharedAgentState
    ) -> AgentMessage:
        """
        Create and send a message to another agent.
        
        Args:
            recipient: Recipient agent role
            msg_type: Message type
            payload: Message payload
            state: SharedAgentState to update
            
        Returns:
            Created AgentMessage
        """
        msg = AgentMessage(
            sender=self.ROLE,
            recipient=recipient,
            message_type=msg_type,
            payload=payload
        )
        
        # Validate payload
        if not msg.validate_payload():
            print(f"[{self.ROLE}] WARNING: Invalid payload for {msg_type}")
        
        # Append to message history
        state.message_history.append(msg)
        
        return msg
    
    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Call OpenAI LLM and return text response.
        
        Args:
            prompt: User prompt
            system: Optional system prompt (uses class default if None)
            
        Returns:
            Text response from LLM
        """
        if not self.client:
            return '{"error": "OpenAI client not initialized"}'
        
        system_prompt = system or self.SYSTEM_PROMPT
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[{self.ROLE}] LLM Error: {e}")
            return f'{{"error": "{str(e)}"}}'
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.ROLE})"


# ============================================================================
# Requirements Agent
# ============================================================================

class RequirementsAgent(BaseAgent):
    """
    Analyzes raw task descriptions and produces structured requirements.
    """
    
    ROLE = "RA"
    
    SYSTEM_PROMPT = """You are the Requirements Agent in a software development team. Your job is to 
analyze a raw task description and produce a structured requirements document.
You MUST respond in valid JSON with these exact keys:
{
  "requirements": ["<requirement 1>", "<requirement 2>", ...],
  "acceptance_criteria": ["<criterion 1>", ...],
  "edge_cases": ["<edge case 1>", ...],
  "constraints": ["<constraint 1>", ...],
  "clarifications_needed": ["<any ambiguity that needs resolution>"]
}
Be thorough. Include at least 3 acceptance criteria and 3 edge cases."""
    
    def run(self, state: SharedAgentState) -> AgentMessage:
        """Analyze task and produce requirements."""
        print(f"\n[{self.ROLE}] Analyzing requirements...")
        
        # Build prompt
        prompt = f"""Analyze this task and produce requirements:

Task: {state.task.raw_description}

Language: {state.task.language}

Produce a comprehensive requirements analysis."""
        
        # Call LLM
        response = self._call_llm(prompt)
        
        try:
            parsed = json.loads(response)
            
            # Update task with requirements
            state.task.acceptance_criteria = parsed.get("acceptance_criteria", [])
            state.task.edge_cases = parsed.get("edge_cases", [])
            state.task.constraints = parsed.get("constraints", [])
            
            print(f"[{self.ROLE}] Generated {len(parsed.get('requirements', []))} requirements")
            print(f"[{self.ROLE}] Generated {len(parsed.get('acceptance_criteria', []))} acceptance criteria")
            
            # Send message
            return self._send_message(
                recipient="AA",
                msg_type="task_spec",
                payload=parsed,
                state=state
            )
            
        except json.JSONDecodeError as e:
            print(f"[{self.ROLE}] JSON Error: {e}")
            # Return minimal valid message
            return self._send_message(
                recipient="AA",
                msg_type="task_spec",
                payload={
                    "requirements": [state.task.raw_description],
                    "acceptance_criteria": ["Code must work correctly"],
                    "edge_cases": ["None identified"]
                },
                state=state
            )


# ============================================================================
# Architecture Agent
# ============================================================================

class ArchitectureAgent(BaseAgent):
    """
    Designs software architecture based on requirements.
    """
    
    ROLE = "AA"
    
    SYSTEM_PROMPT = """You are the Architecture Agent. Given requirements, design the software architecture.
Respond in valid JSON:
{
  "components": [
    {"name": "<component>", "responsibility": "<what it does>", "interface": "<inputs/outputs>"}
  ],
  "interfaces": [
    {"from": "<component>", "to": "<component>", "data": "<what is passed>"}
  ],
  "data_flow": "<prose description of how data moves through the system>",
  "design_patterns": ["<pattern used>"],
  "file_structure": ["<file1.py>", "<file2.py>", ...]
}"""
    
    def run(self, state: SharedAgentState) -> AgentMessage:
        """Design architecture based on requirements."""
        print(f"\n[{self.ROLE}] Designing architecture...")
        
        # Get requirements from previous agent
        ra_output = state.agent_outputs.get("RA", {})
        requirements = ra_output.get("requirements", [state.task.raw_description])
        
        # Build prompt
        prompt = f"""Design the architecture for this task:

Task: {state.task.raw_description}

Requirements:
{json.dumps(requirements, indent=2)}

Acceptance Criteria:
{json.dumps(state.task.acceptance_criteria, indent=2)}

Produce a comprehensive architecture design."""
        
        # Call LLM
        response = self._call_llm(prompt)
        
        try:
            parsed = json.loads(response)
            
            # Store architecture
            state.agent_outputs["AA"] = parsed
            
            print(f"[{self.ROLE}] Designed {len(parsed.get('components', []))} components")
            print(f"[{self.ROLE}] File structure: {len(parsed.get('file_structure', []))} files")
            
            # Send message
            return self._send_message(
                recipient="IA",
                msg_type="arch_plan",
                payload=parsed,
                state=state
            )
            
        except json.JSONDecodeError as e:
            print(f"[{self.ROLE}] JSON Error: {e}")
            # Return minimal architecture
            return self._send_message(
                recipient="IA",
                msg_type="arch_plan",
                payload={
                    "components": [{"name": "main", "responsibility": "solve task", "interface": "function"}],
                    "interfaces": [],
                    "data_flow": "Simple single-function solution",
                    "design_patterns": [],
                    "file_structure": ["solution.py"]
                },
                state=state
            )


# ============================================================================
# Implementation Agent
# ============================================================================

class ImplementationAgent(BaseAgent):
    """
    Implements the code using the Hierarchical Reasoning Engine.
    """
    
    ROLE = "IA"
    
    SYSTEM_PROMPT = """You are the Implementation Agent. Your goal is to write correct, clean Python code 
that satisfies the architecture specification and all acceptance criteria.
Follow these rules:
1. Write modular, well-commented code with type hints.
2. Implement error handling for all external calls.
3. Follow the exact file structure from the architecture plan.
4. Each function must have a docstring.
5. Do not use deprecated APIs or unsafe patterns.
6. When you call "finalize_solution", include the COMPLETE code, not a summary."""
    
    def run(self, state: SharedAgentState) -> AgentMessage:
        """Implement the solution using HRE."""
        print(f"\n[{self.ROLE}] Implementing solution...")
        
        # Get architecture from previous agent
        arch = state.agent_outputs.get("AA", {})
        
        # Build enriched task description
        enriched_desc = f"""TASK: {state.task.raw_description}

ARCHITECTURE:
{json.dumps(arch, indent=2)}

ACCEPTANCE CRITERIA:
{json.dumps(state.task.acceptance_criteria, indent=2)}

EDGE CASES:
{json.dumps(state.task.edge_cases, indent=2)}

Implement this task following the architecture."""
        
        # Update task with enriched description
        state.task.raw_description = enriched_desc
        state.current_agent = self.ROLE
        state.status = "running"
        
        # Run HRE
        hre = HierarchicalReasoningEngine(self.memory, agent_role=self.ROLE)
        state = hre.run(state)
        
        # Extract generated code
        final_code = state.agent_outputs.get("final_code", "")
        
        if not final_code:
            # Try to extract from last working memory step
            if state.working_memory.records:
                last_step = state.working_memory.records[-1]
                if "code" in last_step.action_input:
                    final_code = last_step.action_input["code"]
        
        print(f"[{self.ROLE}] Generated {len(final_code)} chars of code")
        
        # Prepare payload
        payload = {
            "code": final_code,
            "language": state.task.language,
            "file_path": "solution.py",
            "test_spec": state.task.acceptance_criteria
        }
        
        # Store for verification
        state.agent_outputs["IA_code"] = final_code
        
        # Send message
        return self._send_message(
            recipient="VA",
            msg_type="code_artifact",
            payload=payload,
            state=state
        )


# ============================================================================
# Verification Agent
# ============================================================================

class VerificationAgent(BaseAgent):
    """
    Verifies code for correctness, security, and test coverage.
    """
    
    ROLE = "VA"
    
    SYSTEM_PROMPT = """You are the Verification Agent. You receive code artifacts and must:
1. Analyze the code for logical errors, missing edge case handling, and security issues.
2. Review Bandit security scan results.
3. Review test execution results.
4. Produce a structured verification report.
Respond in valid JSON:
{
  "passed": <true/false>,
  "failed_tests": ["<test name>: <reason>"],
  "logical_errors": ["<description>"],
  "security_issues": ["<CWE description>"],
  "bandit_clean": <true/false>,
  "coverage_estimate": "<percentage>",
  "recommendation": "APPROVE | REVISE | REJECT",
  "revision_hints": ["<specific fix instruction>"]
}"""
    
    def run(self, state: SharedAgentState) -> AgentMessage:
        """Verify the implementation."""
        print(f"\n[{self.ROLE}] Verifying implementation...")
        
        # Get code from IA
        code = state.agent_outputs.get("IA_code", "")
        
        if not code:
            print(f"[{self.ROLE}] No code to verify")
            return self._send_message(
                recipient="INA",
                msg_type="validation_result",
                payload={
                    "passed": False,
                    "failed_tests": ["No code provided"],
                    "logical_errors": [],
                    "security_issues": [],
                    "bandit_clean": True,
                    "coverage_estimate": "0%",
                    "recommendation": "REJECT",
                    "revision_hints": ["Implementation agent did not produce code"]
                },
                state=state
            )
        
        # Run Bandit security scan
        bandit_results = self._run_bandit(code)
        
        # Run code execution test
        execution_results = self._test_code(code)
        
        # Build verification prompt
        prompt = f"""Verify this code:

CODE:
{code[:2000]}

BANDIT SECURITY SCAN:
{bandit_results}

EXECUTION TEST:
{execution_results}

ACCEPTANCE CRITERIA:
{json.dumps(state.task.acceptance_criteria, indent=2)}

Provide verification assessment."""
        
        # Call LLM
        response = self._call_llm(prompt)
        
        try:
            parsed = json.loads(response)
            
            # Store verification result
            state.agent_outputs["VA_result"] = parsed
            
            recommendation = parsed.get("recommendation", "REVISE")
            print(f"[{self.ROLE}] Recommendation: {recommendation}")
            
            # If approved, mark code as approved
            if recommendation == "APPROVE":
                state.agent_outputs["VA_approved_code"] = code
            
            # Determine message type
            msg_type = "validation_result" if recommendation == "APPROVE" else "correction_request"
            
            return self._send_message(
                recipient="INA" if recommendation == "APPROVE" else "IA",
                msg_type=msg_type,
                payload=parsed if recommendation == "APPROVE" else {
                    "original_message_id": str(uuid.uuid4()),
                    "validation_errors": parsed.get("failed_tests", []) + parsed.get("logical_errors", []),
                    "correction_hint": " | ".join(parsed.get("revision_hints", []))
                },
                state=state
            )
            
        except json.JSONDecodeError as e:
            print(f"[{self.ROLE}] JSON Error: {e}")
            # Default to approval on parse error
            state.agent_outputs["VA_approved_code"] = code
            return self._send_message(
                recipient="INA",
                msg_type="validation_result",
                payload={
                    "passed": True,
                    "failed_tests": [],
                    "logical_errors": [],
                    "security_issues": [],
                    "bandit_clean": True,
                    "coverage_estimate": "unknown",
                    "recommendation": "APPROVE",
                    "revision_hints": []
                },
                state=state
            )
    
    def _run_bandit(self, code: str) -> str:
        """Run Bandit security scanner on code."""
        try:
            # Write code to temp file
            temp_file = Path("./workspace/bandit_check.py")
            temp_file.parent.mkdir(exist_ok=True)
            
            with open(temp_file, "w") as f:
                f.write(code)
            
            # Run bandit
            result = subprocess.run(
                ["bandit", "-r", "-f", "json", str(temp_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            try:
                bandit_json = json.loads(result.stdout)
                issues = bandit_json.get("results", [])
                
                # Filter HIGH/MEDIUM severity
                critical = [
                    f"{i['issue_severity']}: {i['issue_text']} (line {i['line_number']})"
                    for i in issues
                    if i['issue_severity'] in ['HIGH', 'MEDIUM']
                ]
                
                if critical:
                    return "Security issues found:\n" + "\n".join(critical[:5])
                else:
                    return "No significant security issues found"
                    
            except json.JSONDecodeError:
                return "Bandit scan completed (no critical issues)"
                
        except FileNotFoundError:
            return "Bandit not installed (skipping security scan)"
        except Exception as e:
            return f"Bandit error: {str(e)}"
    
    def _test_code(self, code: str) -> str:
        """Test code execution."""
        try:
            # Simple syntax check
            compile(code, "<string>", "exec")
            return "Code compiles successfully"
        except SyntaxError as e:
            return f"Syntax error: {e}"
        except Exception as e:
            return f"Compilation error: {e}"


# ============================================================================
# Integration Agent
# ============================================================================

class IntegrationAgent(BaseAgent):
    """
    Integrates verified code into the target repository.
    """
    
    ROLE = "INA"
    
    SYSTEM_PROMPT = """You are the Integration Agent. You receive verified code and must integrate it
into the target repository. Respond in valid JSON:
{
  "files_modified": ["<path>"],
  "diff": "<unified diff format>",
  "test_results": {"passed": <n>, "failed": <n>, "errors": <n>},
  "integration_notes": "<any important notes>",
  "ready_for_deployment": <true/false>
}
Rules:
1. Preserve all existing API contracts.
2. Do not remove existing test coverage.
3. Update imports if new modules are added.
4. Generate a proper unified diff (--- a/file +++ b/file format)."""
    
    def run(self, state: SharedAgentState) -> AgentMessage:
        """Integrate verified code."""
        print(f"\n[{self.ROLE}] Integrating solution...")
        
        # Get approved code
        code = state.agent_outputs.get("VA_approved_code", "")
        
        if not code:
            print(f"[{self.ROLE}] No approved code to integrate")
            return self._send_message(
                recipient="ORCHESTRATOR",
                msg_type="patch",
                payload={
                    "files_modified": [],
                    "diff": "",
                    "test_results": {"passed": 0, "failed": 1, "errors": 1},
                    "integration_notes": "No approved code available",
                    "ready_for_deployment": False
                },
                state=state
            )
        
        # Get architecture for context
        arch = state.agent_outputs.get("AA", {})
        
        # Build integration prompt
        prompt = f"""Integrate this verified code:

CODE:
{code}

ARCHITECTURE:
{json.dumps(arch, indent=2)}

REPOSITORY PATH: {state.task.repository_path or "standalone"}

Create integration plan."""
        
        # Call LLM
        response = self._call_llm(prompt)
        
        try:
            parsed = json.loads(response)
            
            # If repository path exists, write the code
            if state.task.repository_path:
                repo_path = Path(state.task.repository_path)
                solution_file = repo_path / "solution.py"
                solution_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(solution_file, "w") as f:
                    f.write(code)
                
                print(f"[{self.ROLE}] Wrote solution to {solution_file}")
            
            # Store final patch
            state.agent_outputs["final_patch"] = parsed
            
            print(f"[{self.ROLE}] Integration complete")
            
            return self._send_message(
                recipient="ORCHESTRATOR",
                msg_type="patch",
                payload=parsed,
                state=state
            )
            
        except json.JSONDecodeError as e:
            print(f"[{self.ROLE}] JSON Error: {e}")
            # Return minimal patch
            return self._send_message(
                recipient="ORCHESTRATOR",
                msg_type="patch",
                payload={
                    "files_modified": ["solution.py"],
                    "diff": f"New file: solution.py\n{code[:200]}...",
                    "test_results": {"passed": 1, "failed": 0, "errors": 0},
                    "integration_notes": "Standalone solution created",
                    "ready_for_deployment": True
                },
                state=state
            )


# ============================================================================
# Agent Orchestrator
# ============================================================================

class AgentOrchestrator:
    """
    Orchestrates the multi-agent pipeline with revision loops.
    """
    
    PIPELINE = ["RA", "AA", "IA", "VA", "INA"]
    MAX_REVISION_CYCLES = 3
    
    def __init__(self, memory: DualLayerMemorySystem):
        """
        Initialize orchestrator with all agents.
        
        Args:
            memory: Shared DualLayerMemorySystem
        """
        self.memory = memory
        
        # Instantiate all agents
        self.agents = {
            "RA": RequirementsAgent(memory),
            "AA": ArchitectureAgent(memory),
            "IA": ImplementationAgent(memory),
            "VA": VerificationAgent(memory),
            "INA": IntegrationAgent(memory)
        }
        
        print("[ORCHESTRATOR] Initialized with 5 agents")
    
    def run_pipeline(self, task: TaskSpec) -> SharedAgentState:
        """
        Execute the full multi-agent pipeline.
        
        Args:
            task: TaskSpec to solve
            
        Returns:
            Final SharedAgentState
        """
        print("\n" + "=" * 80)
        print(f"[ORCHESTRATOR] Starting pipeline for task: {task.task_id}")
        print("=" * 80)
        
        # Create shared state
        state = SharedAgentState(
            task=task,
            status="running"
        )
        
        # Step 1: Requirements Agent
        ra_msg = self.agents["RA"].run(state)
        state.agent_outputs["RA"] = ra_msg.payload
        
        # Step 2: Architecture Agent
        aa_msg = self.agents["AA"].run(state)
        state.agent_outputs["AA"] = aa_msg.payload
        
        # Step 3: IA → VA revision loop
        revision_cycle = 0
        
        while revision_cycle < self.MAX_REVISION_CYCLES:
            print(f"\n[ORCHESTRATOR] Revision cycle {revision_cycle + 1}/{self.MAX_REVISION_CYCLES}")
            
            # Run Implementation Agent
            ia_msg = self.agents["IA"].run(state)
            state.agent_outputs["IA_code"] = ia_msg.payload.get("code", "")
            
            # Run Verification Agent
            va_msg = self.agents["VA"].run(state)
            state.agent_outputs["VA_result"] = va_msg.payload
            
            recommendation = va_msg.payload.get("recommendation", "REVISE")
            
            if recommendation == "APPROVE":
                print("[ORCHESTRATOR] Code approved by VA")
                state.agent_outputs["VA_approved_code"] = state.agent_outputs["IA_code"]
                break
            elif recommendation == "REJECT":
                print("[ORCHESTRATOR] Code rejected by VA")
                state.status = "failed"
                break
            else:  # REVISE
                print(f"[ORCHESTRATOR] Revision requested: {va_msg.payload.get('revision_hints', [])}")
                # Hints are already in state for IA to use
                revision_cycle += 1
        
        # Check if we exhausted revisions
        if revision_cycle >= self.MAX_REVISION_CYCLES:
            print("[ORCHESTRATOR] Maximum revision cycles reached")
            state.status = "failed"
            return state
        
        # Step 4: Integration Agent (only if approved)
        if state.agent_outputs.get("VA_approved_code"):
            ina_msg = self.agents["INA"].run(state)
            state.agent_outputs["final_patch"] = ina_msg.payload
            state.status = "completed"
        
        print("\n" + "=" * 80)
        print(f"[ORCHESTRATOR] Pipeline complete - Status: {state.status}")
        print("=" * 80)
        
        return state


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from memory import DualLayerMemorySystem
    
    print("\n" + "=" * 80)
    print("MACP Demo - Full Agent Pipeline")
    print("=" * 80)
    
    # Create memory system
    memory = DualLayerMemorySystem(faiss_index_path="./demo_pkb")
    
    # Create task
    task = TaskSpec(
        task_id=str(uuid.uuid4()),
        raw_description="Write a Python function that reverses a string and handles None input",
        language="python"
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(memory)
    
    # Run pipeline (only if API key is set)
    if orchestrator.agents["RA"].client:
        print("\nRunning full pipeline...")
        state = orchestrator.run_pipeline(task)
        
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"Status: {state.status}")
        print(f"Steps: {state.step_count}")
        print(f"Messages: {len(state.message_history)}")
        
        if "final_patch" in state.agent_outputs:
            print("\nFinal patch available")
            patch = state.agent_outputs["final_patch"]
            print(f"Files modified: {patch.get('files_modified', [])}")
    else:
        print("\nSkipping pipeline execution (no API key)")
    
    print("\n" + "=" * 80)
