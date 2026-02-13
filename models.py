"""
Core data models for the Autonomous Agent Framework (AAF).

This module defines all Pydantic v2 models used across the three main modules:
- Hierarchical Reasoning Engine (HRE)
- Dual-Layer Adaptive Memory System (DLAMS)
- Multi-Agent Coordination Protocol (MACP)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasoningStep(BaseModel):
    """
    Represents a single step in the ReAct reasoning loop.
    
    Each step captures the agent's thought process, the action taken,
    the result of that action, and metadata for memory management.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    step_id: int = Field(..., description="Sequential step identifier")
    thought: str = Field(..., description="Agent's explicit reasoning trace")
    action: str = Field(..., description="Tool name or agent action to execute")
    action_input: dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    observation: str = Field(default="", description="Result from tool execution")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score (0.0-1.0)")
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for working memory eviction")
    is_error: bool = Field(default=False, description="True if observation contains an error")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v
    
    def __repr__(self) -> str:
        return f"ReasoningStep(id={self.step_id}, action={self.action}, confidence={self.confidence:.2f})"


class WorkingMemoryRecord(BaseModel):
    """
    Sliding window working memory with priority-based eviction.
    
    Maintains the most recent and important reasoning steps, evicting
    lower-priority items when the maximum size is reached.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    records: list[ReasoningStep] = Field(default_factory=list, description="List of reasoning steps")
    max_size: int = Field(default=12, description="Maximum number of records to maintain")
    
    def add(self, step: ReasoningStep) -> None:
        """
        Add a step to working memory, evicting lowest-priority step if full.
        
        Args:
            step: ReasoningStep to add to memory
        """
        if len(self.records) >= self.max_size:
            # Find and remove the lowest-priority record
            min_priority_idx = min(range(len(self.records)), 
                                  key=lambda i: self.records[i].priority)
            self.records.pop(min_priority_idx)
        
        self.records.append(step)
    
    def to_prompt_string(self) -> str:
        """
        Format records as a string for LLM prompt injection.
        
        Returns:
            Formatted string representation of working memory
        """
        if not self.records:
            return "Working memory is empty."
        
        lines = []
        for record in self.records:
            lines.append(f"Step {record.step_id}:")
            lines.append(f"  Thought: {record.thought[:200]}...")
            lines.append(f"  Action: {record.action}")
            lines.append(f"  Observation: {record.observation[:200]}...")
            lines.append(f"  Confidence: {record.confidence:.2f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all records from working memory."""
        self.records.clear()


class PKBEntry(BaseModel):
    """
    Persistent Knowledge Base entry for long-term storage.
    
    Stores code solutions, API documentation, constraints, and error patterns
    with vector embeddings for semantic retrieval.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique entry identifier")
    task_fingerprint: str = Field(..., description="Normalized hash of the task")
    content: str = Field(..., description="Code snippet, documentation, or specification")
    embedding: list[float] = Field(default_factory=list, description="3072-dim vector (optional at creation)")
    entry_type: Literal["code_solution", "api_doc", "constraint", "error_pattern"] = Field(..., description="Type of knowledge")
    created_at: datetime = Field(default_factory=datetime.now)
    source_task: str = Field(..., description="Original task that generated this entry")
    
    @field_validator('task_fingerprint')
    @classmethod
    def validate_fingerprint(cls, v: str) -> str:
        """Ensure task fingerprint is not empty."""
        if not v or not v.strip():
            raise ValueError("Task fingerprint cannot be empty")
        return v.strip()


class TaskSpec(BaseModel):
    """
    Complete specification for a software development task.
    
    Includes the original description, constraints, acceptance criteria,
    and optional repository paths for SWE-bench style tasks.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    raw_description: str = Field(..., description="Original user or benchmark input")
    language: str = Field(default="python", description="Programming language")
    repository_path: Optional[str] = Field(default=None, description="Path to repository (for SWE-bench)")
    test_file_path: Optional[str] = Field(default=None, description="Path to test file")
    acceptance_criteria: list[str] = Field(default_factory=list, description="Requirements for acceptance")
    edge_cases: list[str] = Field(default_factory=list, description="Edge cases to handle")
    constraints: list[str] = Field(default_factory=list, description="Implementation constraints")
    
    @field_validator('raw_description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure task description is not empty."""
        if not v or not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()


class AgentMessage(BaseModel):
    """
    Structured message for inter-agent communication in MACP.
    
    Each message has a type-specific payload schema that is validated
    to ensure proper data flow between agents.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Payload schema definitions for each message type
    PAYLOAD_SCHEMAS: dict[str, list[str]] = {
        "task_spec": ["requirements", "acceptance_criteria", "edge_cases"],
        "arch_plan": ["components", "interfaces", "data_flow"],
        "code_artifact": ["code", "language", "file_path", "test_spec"],
        "error_report": ["errors", "line_numbers", "severity", "bandit_findings"],
        "patch": ["files_modified", "diff", "test_results"],
        "validation_result": ["passed", "failed_tests", "coverage", "bandit_clean"],
        "correction_request": ["original_message_id", "validation_errors", "correction_hint"]
    }
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message identifier")
    sender: Literal["RA", "AA", "IA", "VA", "INA", "ORCHESTRATOR"] = Field(..., description="Sending agent")
    recipient: Literal["RA", "AA", "IA", "VA", "INA", "ORCHESTRATOR"] = Field(..., description="Receiving agent")
    message_type: Literal[
        "task_spec", "arch_plan", "code_artifact", "error_report",
        "patch", "validation_result", "correction_request"
    ] = Field(..., description="Type of message")
    payload: dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)
    schema_version: str = Field(default="1.0", description="Schema version")
    
    def validate_payload(self) -> bool:
        """
        Validate that payload contains all required keys for the message type.
        
        Returns:
            True if payload is valid, False otherwise
        """
        required_keys = self.PAYLOAD_SCHEMAS.get(self.message_type, [])
        return all(key in self.payload for key in required_keys)
    
    def __repr__(self) -> str:
        return f"AgentMessage({self.sender}→{self.recipient}, type={self.message_type})"


class SharedAgentState(BaseModel):
    """
    Shared state across all agents in the MACP orchestration.
    
    Maintains the current task, working memory, agent outputs, and
    execution status. Provides audit logging capabilities.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    task: TaskSpec = Field(..., description="Current task specification")
    working_memory: WorkingMemoryRecord = Field(default_factory=WorkingMemoryRecord)
    current_agent: str = Field(default="ORCHESTRATOR", description="Currently active agent")
    agent_outputs: dict[str, Any] = Field(default_factory=dict, description="Outputs keyed by agent name")
    message_history: list[AgentMessage] = Field(default_factory=list, description="All inter-agent messages")
    step_count: int = Field(default=0, description="Number of reasoning steps taken")
    max_steps: int = Field(default=50, description="Maximum allowed steps")
    status: Literal["pending", "running", "completed", "failed", "safety_blocked"] = Field(
        default="pending", description="Current execution status"
    )
    audit_log_path: str = Field(default="", description="Path to audit log file")
    
    def log_step(self, step: ReasoningStep) -> None:
        """
        Append a reasoning step to the JSONL audit log file.
        
        Args:
            step: ReasoningStep to log
        """
        if not self.audit_log_path:
            # Initialize audit log path if not set
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.audit_log_path = str(log_dir / f"audit_{self.session_id}.jsonl")
        
        with open(self.audit_log_path, "a") as f:
            log_entry = {
                "session_id": self.session_id,
                "step_id": step.step_id,
                "timestamp": step.timestamp.isoformat(),
                "agent": self.current_agent,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "confidence": step.confidence,
                "priority": step.priority,
                "is_error": step.is_error
            }
            f.write(json.dumps(log_entry) + "\n")
    
    def is_step_limit_reached(self) -> bool:
        """
        Check if the maximum step limit has been reached.
        
        Returns:
            True if step limit reached, False otherwise
        """
        return self.step_count >= self.max_steps
    
    def get_context_summary(self) -> str:
        """
        Get a readable summary of the last 3 agent messages.
        
        Returns:
            Formatted string with recent message context
        """
        if not self.message_history:
            return "No messages in history."
        
        recent_messages = self.message_history[-3:]
        lines = ["Recent Agent Messages:"]
        for msg in recent_messages:
            lines.append(f"  {msg.sender} → {msg.recipient}: {msg.message_type}")
            lines.append(f"    Payload keys: {list(msg.payload.keys())}")
        
        return "\n".join(lines)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AAF Core Data Models - Usage Examples")
    print("=" * 80)
    
    # Example 1: ReasoningStep
    print("\n1. ReasoningStep Example:")
    step = ReasoningStep(
        step_id=1,
        thought="I need to write a function to reverse a string",
        action="write_code",
        action_input={"code": "def reverse(s): return s[::-1]", "file_path": "solution.py"},
        confidence=0.85,
        priority=1.0
    )
    print(step)
    print(f"   Is error: {step.is_error}")
    
    # Example 2: WorkingMemoryRecord
    print("\n2. WorkingMemoryRecord Example:")
    wm = WorkingMemoryRecord(max_size=3)
    for i in range(5):
        wm.add(ReasoningStep(
            step_id=i,
            thought=f"Thought {i}",
            action="test_action",
            confidence=0.7 + i * 0.05,
            priority=0.5 if i < 3 else 1.0  # Last two have higher priority
        ))
    print(f"   Records in memory: {len(wm.records)}")
    print(f"   Step IDs: {[r.step_id for r in wm.records]}")
    
    # Example 3: PKBEntry
    print("\n3. PKBEntry Example:")
    entry = PKBEntry(
        task_fingerprint="reverse_string_hash",
        content="def reverse(s): return s[::-1]",
        entry_type="code_solution",
        source_task="Reverse a string"
    )
    print(f"   Entry ID: {entry.entry_id}")
    print(f"   Type: {entry.entry_type}")
    
    # Example 4: TaskSpec
    print("\n4. TaskSpec Example:")
    task = TaskSpec(
        raw_description="Write a function to check if a number is prime",
        language="python",
        acceptance_criteria=["Handles edge cases", "Efficient algorithm"],
        edge_cases=["n=0", "n=1", "n=2", "negative numbers"]
    )
    print(f"   Task ID: {task.task_id}")
    print(f"   Criteria: {len(task.acceptance_criteria)}")
    
    # Example 5: AgentMessage
    print("\n5. AgentMessage Example:")
    msg = AgentMessage(
        sender="RA",
        recipient="AA",
        message_type="task_spec",
        payload={
            "requirements": ["Implement prime check"],
            "acceptance_criteria": ["Handles edge cases"],
            "edge_cases": ["n=0", "n=1"]
        }
    )
    print(msg)
    print(f"   Valid payload: {msg.validate_payload()}")
    
    # Example 6: SharedAgentState
    print("\n6. SharedAgentState Example:")
    state = SharedAgentState(
        task=task,
        current_agent="IA",
        max_steps=10
    )
    state.step_count = 5
    print(f"   Session ID: {state.session_id}")
    print(f"   Step limit reached: {state.is_step_limit_reached()}")
    print(f"   Status: {state.status}")
    
    # Test audit logging
    state.log_step(step)
    print(f"   Audit log created: {state.audit_log_path}")
    
    print("\n" + "=" * 80)
    print("All models validated successfully!")
    print("=" * 80)
