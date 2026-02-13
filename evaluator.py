"""
Evaluation harness for the AAF system.

Supports HumanEval benchmark with built-in sample problems.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track
from rich.table import Table

from agents import AgentOrchestrator
from memory import DualLayerMemorySystem
from models import SharedAgentState, TaskSpec


class AAFEvaluator:
    """
    Evaluation harness for AAF on code generation benchmarks.
    """
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory and orchestrator
        self.memory = DualLayerMemorySystem(faiss_index_path="./data/eval_pkb")
        self.orchestrator = AgentOrchestrator(self.memory)
        
        self.console = Console()
        
        self.console.print("[bold green]AAF Evaluator initialized[/bold green]")
    
    def load_humaneval(self, path: str = None) -> list[dict]:
        """
        Load HumanEval problems.
        
        Args:
            path: Optional path to HumanEval JSON file
            
        Returns:
            List of problem dicts
        """
        if path and Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
        
        # Built-in sample problems
        problems = [
            {
                "task_id": "HE/1",
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
                "canonical_solution": "    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i]-numbers[j]) < threshold:\n                return True\n    return False",
                "test": "from typing import List\nassert has_close_elements([1.0,2.0,3.9,4.0,5.0,2.2],0.3)==True\nassert has_close_elements([1.0,2.0,5.9,4.0,5.0],0.3)==False\nassert has_close_elements([1.0,2.0,3.0,4.0,5.0,2.0],0.3)==True"
            },
            {
                "task_id": "HE/2",
                "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Separate parenthesis groups.\"\"\"\n",
                "canonical_solution": "    result=[]\n    current=''\n    depth=0\n    for c in paren_string.replace(' ',''):\n        current+=c\n        if c=='(': depth+=1\n        elif c==')':\n            depth-=1\n            if depth==0:\n                result.append(current)\n                current=''\n    return result",
                "test": "from typing import List\nassert separate_paren_groups('( ) (( )) (( )( ))')==['()','(())','(()())']"
            },
            {
                "task_id": "HE/3",
                "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return decimal part of float.\"\"\"\n",
                "canonical_solution": "    return number % 1.0",
                "test": "assert truncate_number(3.5)==0.5\nassert abs(truncate_number(1.33)-0.33)<0.01"
            },
            {
                "task_id": "HE/4",
                "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\"Check if balance goes below zero.\"\"\"\n",
                "canonical_solution": "    balance=0\n    for op in operations:\n        balance+=op\n        if balance<0: return True\n    return False",
                "test": "from typing import List\nassert below_zero([1,2,3])==False\nassert below_zero([1,2,-4,5])==True\nassert below_zero([1,2,-3])==False"
            },
            {
                "task_id": "HE/5",
                "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"Mean absolute deviation from mean.\"\"\"\n",
                "canonical_solution": "    mean=sum(numbers)/len(numbers)\n    return sum(abs(x-mean) for x in numbers)/len(numbers)",
                "test": "from typing import List\nassert abs(mean_absolute_deviation([1.0,2.0,3.0])-0.6667)<0.001\nassert abs(mean_absolute_deviation([1.0,2.0,3.0,4.0])-1.0)<0.001"
            },
            {
                "task_id": "HE/6",
                "prompt": "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\"Intersperse delimeter between elements.\"\"\"\n",
                "canonical_solution": "    if not numbers: return []\n    result=[]\n    for n in numbers[:-1]:\n        result+=[n,delimeter]\n    return result+[numbers[-1]]",
                "test": "from typing import List\nassert intersperse([],4)==[]\nassert intersperse([1,2,3],4)==[1,4,2,4,3]\nassert intersperse([1],0)==[1]"
            },
            {
                "task_id": "HE/7",
                "prompt": "def parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\"Max nesting depth per group.\"\"\"\n",
                "canonical_solution": "    def max_depth(s):\n        depth=max_d=0\n        for c in s:\n            if c=='(': depth+=1; max_d=max(max_d,depth)\n            elif c==')': depth-=1\n        return max_d\n    return [max_depth(g) for g in paren_string.split()]",
                "test": "from typing import List\nassert parse_nested_parens('(()()) ((())) () ((())(()))')==[2,3,1,3]"
            },
            {
                "task_id": "HE/8",
                "prompt": "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\"Filter strings containing substring.\"\"\"\n",
                "canonical_solution": "    return [s for s in strings if substring in s]",
                "test": "from typing import List\nassert filter_by_substring([],'a')==[]\nassert filter_by_substring(['abc','bcd','cde'],'bc')==['abc','bcd']\nassert filter_by_substring(['abc','def'],'z')==[]"
            },
            {
                "task_id": "HE/9",
                "prompt": "def sum_product(numbers: List[int]) -> Tuple[int,int]:\n    \"\"\"Return (sum, product) of numbers.\"\"\"\n",
                "canonical_solution": "    s=sum(numbers)\n    p=1\n    for n in numbers: p*=n\n    return (s,p)",
                "test": "from typing import List, Tuple\nassert sum_product([])==(0,1)\nassert sum_product([1,2,3,4])==(10,24)\nassert sum_product([1,1,1])==(3,1)"
            },
            {
                "task_id": "HE/10",
                "prompt": "def rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\"Rolling maximum.\"\"\"\n",
                "canonical_solution": "    result=[]\n    cur_max=None\n    for n in numbers:\n        cur_max=n if cur_max is None else max(cur_max,n)\n        result.append(cur_max)\n    return result",
                "test": "from typing import List\nassert rolling_max([1,2,3,2,3,4,2])==[1,2,3,3,3,4,4]\nassert rolling_max([])==[]"
            }
        ]
        
        return problems
    
    def evaluate_humaneval(
        self,
        problems: list[dict] = None,
        max_problems: int = 10
    ) -> dict[str, Any]:
        """
        Evaluate AAF on HumanEval benchmark.
        
        Args:
            problems: Optional list of problems (loads built-in if None)
            max_problems: Maximum number of problems to evaluate
            
        Returns:
            Dict with evaluation results
        """
        if problems is None:
            problems = self.load_humaneval()
        
        self.console.print(f"\n[bold]Evaluating on {min(max_problems, len(problems))} HumanEval problems[/bold]\n")
        
        results = []
        
        for problem in track(
            problems[:max_problems],
            description="Running evaluations..."
        ):
            task = TaskSpec(
                task_id=problem["task_id"],
                raw_description=problem["prompt"] + "\n\nYou must implement the function above. Only provide the function implementation, no additional code."
            )
            
            start_time = time.time()
            
            try:
                state = self.orchestrator.run_pipeline(task)
                elapsed = time.time() - start_time
                
                # Extract generated code
                generated_code = state.agent_outputs.get(
                    "VA_approved_code",
                    state.agent_outputs.get("IA_code", "")
                )
                
                # Test execution
                passed = self._run_test(generated_code, problem["test"], problem["prompt"])
                
                results.append({
                    "task_id": problem["task_id"],
                    "passed": passed,
                    "steps_used": state.step_count,
                    "status": state.status,
                    "elapsed_seconds": round(elapsed, 2),
                    "code_length": len(generated_code)
                })
                
            except Exception as e:
                self.console.print(f"[red]Error on {problem['task_id']}: {e}[/red]")
                results.append({
                    "task_id": problem["task_id"],
                    "passed": False,
                    "steps_used": 0,
                    "status": "error",
                    "elapsed_seconds": 0,
                    "code_length": 0,
                    "error": str(e)
                })
        
        # Calculate metrics
        total_problems = len(results)
        passed_count = sum(r["passed"] for r in results)
        pass_at_1 = (passed_count / total_problems * 100) if total_problems > 0 else 0
        
        completed_results = [r for r in results if r["steps_used"] > 0]
        mean_steps = statistics.mean(r["steps_used"] for r in completed_results) if completed_results else 0
        mean_time = statistics.mean(r["elapsed_seconds"] for r in completed_results) if completed_results else 0
        
        return {
            "benchmark": "HumanEval",
            "total_problems": total_problems,
            "passed": passed_count,
            "pass_at_1": round(pass_at_1, 2),
            "mean_steps": round(mean_steps, 1),
            "mean_time_sec": round(mean_time, 2),
            "details": results
        }
    
    def _run_test(self, generated_code: str, test_code: str, prompt: str) -> bool:
        """
        Run test on generated code.
        
        Args:
            generated_code: Generated solution
            test_code: Test assertions
            prompt: Function signature/prompt
            
        Returns:
            True if tests pass, False otherwise
        """
        # Create complete test file
        full_code = generated_code + "\n\n" + test_code
        
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_path = f.name
            
            # Execute
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            # Check result
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            self.console.print(f"[yellow]Test execution error: {e}[/yellow]")
            return False
    
    def print_results(self, results: dict) -> None:
        """
        Print formatted evaluation results.
        
        Args:
            results: Results dict from evaluate_humaneval
        """
        self.console.print("\n" + "=" * 80)
        self.console.print("[bold cyan]EVALUATION RESULTS[/bold cyan]")
        self.console.print("=" * 80 + "\n")
        
        # Summary table
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Benchmark", results["benchmark"])
        summary_table.add_row("Total Problems", str(results["total_problems"]))
        summary_table.add_row("Passed", str(results["passed"]))
        summary_table.add_row("Pass@1 (%)", f"{results['pass_at_1']:.2f}%")
        summary_table.add_row("Mean Steps", f"{results['mean_steps']:.1f}")
        summary_table.add_row("Mean Time (s)", f"{results['mean_time_sec']:.2f}")
        
        self.console.print(summary_table)
        
        # Per-task table
        task_table = Table(title="\nPer-Task Results")
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Passed", style="green")
        task_table.add_column("Steps", style="yellow")
        task_table.add_column("Time (s)", style="magenta")
        task_table.add_column("Status", style="blue")
        
        for detail in results["details"]:
            passed_str = "✓" if detail["passed"] else "✗"
            passed_style = "green" if detail["passed"] else "red"
            
            task_table.add_row(
                detail["task_id"],
                f"[{passed_style}]{passed_str}[/{passed_style}]",
                str(detail["steps_used"]),
                f"{detail['elapsed_seconds']:.2f}",
                detail["status"]
            )
        
        self.console.print(task_table)
        self.console.print("\n" + "=" * 80 + "\n")
    
    def save_results(self, results: dict, filename: str = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            results: Results dict
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        self.console.print(f"[green]Results saved to: {filepath}[/green]")
        
        return str(filepath)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    evaluator = AAFEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_humaneval(max_problems=3)
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
