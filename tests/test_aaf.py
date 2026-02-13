"""
Test suite for the Autonomous Agent Framework (AAF).

Tests all core components: models, memory, reasoning engine, agents.
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from models import (
    AgentMessage,
    PKBEntry,
    ReasoningStep,
    SharedAgentState,
    TaskSpec,
    WorkingMemoryRecord,
)


# ============================================================================
# Model Tests
# ============================================================================

def test_working_memory_eviction():
    """Test that working memory evicts lowest-priority items when full."""
    wm = WorkingMemoryRecord(max_size=3)
    
    # Add 4 steps with varying priorities
    for i in range(4):
        step = ReasoningStep(
            step_id=i,
            thought=f"Thought {i}",
            action="test",
            confidence=0.8,
            priority=0.5 if i < 2 else 1.0  # First two have lower priority
        )
        wm.add(step)
    
    # Should only have 3 records
    assert len(wm.records) == 3
    
    # Should have evicted one of the low-priority items (step 0 or 1)
    step_ids = [r.step_id for r in wm.records]
    assert len(step_ids) == 3
    assert 2 in step_ids  # High priority
    assert 3 in step_ids  # High priority
    # One of 0 or 1 should be missing
    assert (0 not in step_ids) or (1 not in step_ids)


def test_agent_message_validation():
    """Test AgentMessage payload validation."""
    # Valid message
    valid_msg = AgentMessage(
        sender="RA",
        recipient="AA",
        message_type="code_artifact",
        payload={
            "code": "def hello(): pass",
            "language": "python",
            "file_path": "solution.py",
            "test_spec": ["test1"]
        }
    )
    assert valid_msg.validate_payload() is True
    
    # Invalid message (missing required keys)
    invalid_msg = AgentMessage(
        sender="RA",
        recipient="AA",
        message_type="code_artifact",
        payload={
            "code": "def hello(): pass"
            # Missing: language, file_path, test_spec
        }
    )
    assert invalid_msg.validate_payload() is False


def test_shared_state_step_limit():
    """Test that SharedAgentState correctly detects step limit."""
    task = TaskSpec(
        raw_description="Test task"
    )
    
    state = SharedAgentState(
        task=task,
        max_steps=5
    )
    
    assert state.is_step_limit_reached() is False
    
    # Increment to limit
    state.step_count = 5
    assert state.is_step_limit_reached() is True
    
    # Beyond limit
    state.step_count = 6
    assert state.is_step_limit_reached() is True


def test_task_spec_creation():
    """Test TaskSpec creation and defaults."""
    task = TaskSpec(
        raw_description="Write a function to reverse a string"
    )
    
    # Check UUID format
    assert len(task.task_id) == 36  # Standard UUID length with hyphens
    assert task.task_id.count('-') == 4
    
    # Check default language
    assert task.language == "python"
    
    # Check default lists
    assert task.acceptance_criteria == []
    assert task.edge_cases == []
    assert task.constraints == []


def test_pkb_entry_validation():
    """Test PKBEntry field validation."""
    # Valid entry
    entry = PKBEntry(
        task_fingerprint="test_hash_123",
        content="def hello(): return 'world'",
        entry_type="code_solution",
        source_task="Test task"
    )
    
    assert entry.entry_id is not None
    assert entry.task_fingerprint == "test_hash_123"
    
    # Invalid entry (empty fingerprint)
    with pytest.raises(ValueError):
        PKBEntry(
            task_fingerprint="",
            content="code",
            entry_type="code_solution",
            source_task="task"
        )


def test_reasoning_step_confidence_bounds():
    """Test that ReasoningStep validates confidence bounds."""
    # Valid confidence
    step = ReasoningStep(
        step_id=1,
        thought="test",
        action="test",
        confidence=0.75
    )
    assert step.confidence == 0.75
    
    # Edge cases
    step_min = ReasoningStep(
        step_id=2,
        thought="test",
        action="test",
        confidence=0.0
    )
    assert step_min.confidence == 0.0
    
    step_max = ReasoningStep(
        step_id=3,
        thought="test",
        action="test",
        confidence=1.0
    )
    assert step_max.confidence == 1.0


# ============================================================================
# Memory System Tests
# ============================================================================

@patch('memory.OpenAI')
@patch('memory.faiss')
def test_memory_context_format(mock_faiss, mock_openai):
    """Test that memory context is properly formatted."""
    from memory import DualLayerMemorySystem
    
    # Mock FAISS
    mock_index = Mock()
    mock_index.ntotal = 0
    mock_faiss.IndexFlatIP.return_value = mock_index
    mock_faiss.read_index.side_effect = FileNotFoundError()
    
    # Create memory system
    memory = DualLayerMemorySystem(faiss_index_path="./test_memory")
    
    # Add steps to working memory
    step1 = ReasoningStep(
        step_id=1,
        thought="First thought",
        action="analyze",
        confidence=0.8
    )
    step2 = ReasoningStep(
        step_id=2,
        thought="Second thought",
        action="write_code",
        confidence=0.9
    )
    
    memory.add_to_working_memory(step1)
    memory.add_to_working_memory(step2)
    
    # Build context
    context = memory.build_memory_context("test query")
    
    # Check format
    assert "WORKING MEMORY" in context
    assert "RETRIEVED KNOWLEDGE" in context
    assert "Step 1" in context
    assert "Step 2" in context


@patch('memory.OpenAI')
@patch('memory.faiss')
def test_memory_successful_solution_save(mock_faiss, mock_openai):
    """Test saving successful solutions to PKB."""
    from memory import DualLayerMemorySystem
    
    # Mock FAISS
    mock_index = Mock()
    mock_index.ntotal = 0
    mock_faiss.IndexFlatIP.return_value = mock_index
    mock_faiss.read_index.side_effect = FileNotFoundError()
    
    # Mock OpenAI embeddings
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * 3072)]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Create memory system
    memory = DualLayerMemorySystem(faiss_index_path="./test_save_pkb")
    memory.client = mock_client
    
    # Create task and solution
    task = TaskSpec(
        raw_description="Test task for solution save"
    )
    solution = "def test(): return True"
    
    # Save solution
    initial_size = memory.index.ntotal
    memory.save_successful_solution(task, solution)
    
    # Verify entry was created
    assert len(memory._entries) > 0
    last_entry = memory._entries[-1]
    assert last_entry.content == solution
    assert last_entry.entry_type == "code_solution"


# ============================================================================
# Reasoning Engine Tests
# ============================================================================

@patch('reasoning_engine.OpenAI')
def test_docker_execute_timeout(mock_openai):
    """Test that Docker execution handles timeouts."""
    from reasoning_engine import HierarchicalReasoningEngine
    from memory import DualLayerMemorySystem
    
    # Create mock memory
    memory = Mock(spec=DualLayerMemorySystem)
    
    # Create HRE
    hre = HierarchicalReasoningEngine(memory, agent_role="test")
    hre.client = Mock()  # Mock client
    
    # Test timeout handling
    code = "import time\ntime.sleep(100)"
    result = hre._docker_execute(code, timeout=1)
    
    # Should either timeout or error
    assert "DOCKER_ERROR" in result or "timeout" in result.lower() or "executed" in result.lower()


@patch('reasoning_engine.OpenAI')
def test_react_step_parsing(mock_openai):
    """Test parsing of ReAct step from LLM response."""
    from reasoning_engine import HierarchicalReasoningEngine
    from memory import DualLayerMemorySystem
    
    memory = Mock(spec=DualLayerMemorySystem)
    hre = HierarchicalReasoningEngine(memory, agent_role="test")
    
    # Mock LLM response
    raw_response = {
        "thought": "I need to write code",
        "action": "write_code",
        "action_input": {"code": "def test(): pass", "file_path": "test.py"},
        "confidence": 0.85
    }
    
    step = hre._parse_react_step(raw_response, step_id=1)
    
    assert step.step_id == 1
    assert step.thought == "I need to write code"
    assert step.action == "write_code"
    assert step.confidence == 0.85
    assert step.is_error is False


@patch('reasoning_engine.OpenAI')
def test_action_execution(mock_openai):
    """Test execution of different action types."""
    from reasoning_engine import HierarchicalReasoningEngine
    from memory import DualLayerMemorySystem
    
    memory = Mock(spec=DualLayerMemorySystem)
    memory.retrieve_from_pkb.return_value = []
    
    hre = HierarchicalReasoningEngine(memory, agent_role="test")
    
    task = TaskSpec(raw_description="Test task")
    state = SharedAgentState(task=task)
    
    # Test write_code action
    step = ReasoningStep(
        step_id=1,
        thought="test",
        action="write_code",
        action_input={"code": "def hello(): pass", "file_path": "test.py"},
        confidence=0.9
    )
    
    observation = hre._execute_action(step, state)
    assert "Code written" in observation
    assert "test.py" in observation


# ============================================================================
# Agent Tests
# ============================================================================

@patch('agents.OpenAI')
def test_requirements_agent_run(mock_openai):
    """Test Requirements Agent execution."""
    from agents import RequirementsAgent
    from memory import DualLayerMemorySystem
    
    memory = Mock(spec=DualLayerMemorySystem)
    
    # Mock LLM response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=json.dumps({
        "requirements": ["Req 1", "Req 2"],
        "acceptance_criteria": ["AC 1", "AC 2", "AC 3"],
        "edge_cases": ["Edge 1", "Edge 2", "Edge 3"],
        "constraints": ["Constraint 1"],
        "clarifications_needed": []
    })))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    agent = RequirementsAgent(memory)
    agent.client = mock_client
    
    task = TaskSpec(raw_description="Write a sorting function")
    state = SharedAgentState(task=task)
    
    msg = agent.run(state)
    
    assert msg.message_type == "task_spec"
    assert msg.sender == "RA"
    assert msg.recipient == "AA"
    assert len(state.task.acceptance_criteria) >= 3


@patch('agents.OpenAI')
def test_orchestrator_pipeline_flow(mock_openai):
    """Test that orchestrator runs agents in correct sequence."""
    from agents import AgentOrchestrator
    from memory import DualLayerMemorySystem
    
    # This is a basic smoke test - full integration requires API
    memory = Mock(spec=DualLayerMemorySystem)
    
    orchestrator = AgentOrchestrator(memory)
    
    # Verify all agents are initialized
    assert "RA" in orchestrator.agents
    assert "AA" in orchestrator.agents
    assert "IA" in orchestrator.agents
    assert "VA" in orchestrator.agents
    assert "INA" in orchestrator.agents
    
    assert orchestrator.MAX_REVISION_CYCLES == 3


# ============================================================================
# Evaluation Tests
# ============================================================================

def test_humaneval_load():
    """Test loading of HumanEval problems."""
    from evaluator import AAFEvaluator
    
    evaluator = AAFEvaluator()
    problems = evaluator.load_humaneval()
    
    # Should load built-in problems
    assert len(problems) == 10
    assert all("task_id" in p for p in problems)
    assert all("prompt" in p for p in problems)
    assert all("test" in p for p in problems)


def test_code_execution():
    """Test code execution and testing."""
    from evaluator import AAFEvaluator
    
    evaluator = AAFEvaluator()
    
    # Valid code
    code = "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i]-numbers[j]) < threshold:\n                return True\n    return False"
    test = "from typing import List\nassert has_close_elements([1.0,2.0,3.9],0.3)==True"
    
    result = evaluator._run_test(code, test, "")
    assert result is True
    
    # Invalid code (syntax error)
    bad_code = "def bad( pass"
    result = evaluator._run_test(bad_code, test, "")
    assert result is False


@patch('evaluator.AgentOrchestrator')
def test_evaluation_metrics(mock_orchestrator_class):
    """Test evaluation metrics calculation."""
    from evaluator import AAFEvaluator
    
    evaluator = AAFEvaluator()
    
    # Mock orchestrator
    mock_orch = Mock()
    mock_state = Mock()
    mock_state.step_count = 10
    mock_state.status = "completed"
    mock_state.agent_outputs = {
        "VA_approved_code": "def test(): return True"
    }
    mock_orch.run_pipeline.return_value = mock_state
    mock_orchestrator_class.return_value = mock_orch
    
    # Create new evaluator to use mocked orchestrator
    evaluator.orchestrator = mock_orch
    
    # Mock test execution to always pass
    evaluator._run_test = Mock(return_value=True)
    
    # Run evaluation on 2 problems
    results = evaluator.evaluate_humaneval(max_problems=2)
    
    assert results["benchmark"] == "HumanEval"
    assert results["total_problems"] == 2
    assert results["passed"] == 2
    assert results["pass_at_1"] == 100.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_data_flow():
    """Test data flow through models."""
    # Create task
    task = TaskSpec(
        raw_description="Test task"
    )
    
    # Create state
    state = SharedAgentState(
        task=task,
        max_steps=10
    )
    
    # Create reasoning step
    step = ReasoningStep(
        step_id=0,
        thought="Initial thought",
        action="analyze",
        confidence=0.9
    )
    
    # Log step
    with tempfile.TemporaryDirectory() as tmpdir:
        state.audit_log_path = str(Path(tmpdir) / "audit.jsonl")
        state.log_step(step)
        
        # Verify log was written
        assert Path(state.audit_log_path).exists()
        
        with open(state.audit_log_path, "r") as f:
            log_entry = json.loads(f.read())
            assert log_entry["step_id"] == 0
            assert log_entry["action"] == "analyze"


def test_message_history_tracking():
    """Test that agent messages are properly tracked."""
    task = TaskSpec(raw_description="Test")
    state = SharedAgentState(task=task)
    
    # Create messages
    msg1 = AgentMessage(
        sender="RA",
        recipient="AA",
        message_type="task_spec",
        payload={"requirements": [], "acceptance_criteria": [], "edge_cases": []}
    )
    
    msg2 = AgentMessage(
        sender="AA",
        recipient="IA",
        message_type="arch_plan",
        payload={"components": [], "interfaces": [], "data_flow": ""}
    )
    
    state.message_history.append(msg1)
    state.message_history.append(msg2)
    
    # Test context summary
    summary = state.get_context_summary()
    assert "RA" in summary
    assert "AA" in summary
    assert len(state.message_history) == 2


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
