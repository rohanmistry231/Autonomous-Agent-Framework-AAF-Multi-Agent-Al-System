"""
Main CLI entry point for the Autonomous Agent Framework (AAF).

Provides commands for:
- solve: Solve a single coding task
- eval: Run benchmark evaluation
- interactive: Interactive REPL mode
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from agents import AgentOrchestrator
from evaluator import AAFEvaluator
from memory import DualLayerMemorySystem
from models import TaskSpec

# Load environment variables
load_dotenv()

# Initialize console
console = Console()


def print_banner():
    """Print ASCII art banner."""
    banner_text = """
╔══════════════════════════════════════════════════════════╗
║   AUTONOMOUS AGENT FRAMEWORK (AAF) v1.0                  ║
║   Research Implementation — Mistry Rohan                 ║
║   Swarrnim Startup and Innovation University             ║
╚══════════════════════════════════════════════════════════╝
"""
    panel = Panel(
        Text(banner_text, style="bold cyan"),
        border_style="bright_blue"
    )
    console.print(panel)


def run_single_task(args):
    """
    Solve a single coding task.
    
    Args:
        args: Command-line arguments
    """
    # Load task
    if args.task:
        task_desc = args.task
    elif args.task_file:
        task_path = Path(args.task_file)
        if not task_path.exists():
            console.print(f"[red]Error: Task file not found: {args.task_file}[/red]")
            return
        
        with open(task_path, "r") as f:
            task_data = json.load(f)
            task_desc = task_data.get("description", task_data.get("prompt", ""))
    else:
        console.print("[red]Error: Must provide --task or --task-file[/red]")
        return
    
    # Create task spec
    task = TaskSpec(
        task_id=str(uuid.uuid4()),
        raw_description=task_desc,
        language="python",
        repository_path=args.repo
    )
    
    console.print(f"\n[bold]Task:[/bold] {task_desc[:100]}...")
    console.print(f"[bold]Task ID:[/bold] {task.task_id}\n")
    
    # Initialize system
    console.print("[yellow]Initializing AAF system...[/yellow]")
    memory = DualLayerMemorySystem(faiss_index_path="./data/pkb_index")
    orchestrator = AgentOrchestrator(memory)
    
    # Run pipeline
    console.print("[yellow]Running agent pipeline...[/yellow]\n")
    
    try:
        state = orchestrator.run_pipeline(task)
        
        # Print results
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]RESULTS[/bold cyan]")
        console.print("=" * 80 + "\n")
        
        # Status
        status_color = "green" if state.status == "completed" else "red"
        console.print(f"[bold]Status:[/bold] [{status_color}]{state.status}[/{status_color}]")
        console.print(f"[bold]Steps Used:[/bold] {state.step_count}")
        console.print(f"[bold]Messages Exchanged:[/bold] {len(state.message_history)}")
        
        # Final code
        if "VA_approved_code" in state.agent_outputs:
            code = state.agent_outputs["VA_approved_code"]
            console.print(f"\n[bold]Final Code ({len(code)} chars):[/bold]")
            
            if args.verbose:
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                console.print(syntax)
            else:
                console.print(f"{code[:200]}...\n(use --verbose to see full code)")
            
            # Save to file if requested
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    f.write(code)
                
                console.print(f"\n[green]Code saved to: {output_path}[/green]")
        
        else:
            console.print("\n[yellow]No approved code generated[/yellow]")
        
        # Audit log
        if state.audit_log_path:
            console.print(f"\n[bold]Audit Log:[/bold] {state.audit_log_path}")
        
        console.print("\n" + "=" * 80 + "\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()


def run_evaluation(args):
    """
    Run benchmark evaluation.
    
    Args:
        args: Command-line arguments
    """
    console.print(f"\n[bold]Running {args.benchmark.upper()} evaluation[/bold]")
    console.print(f"Max problems: {args.max_problems}\n")
    
    # Create evaluator
    evaluator = AAFEvaluator(results_dir=args.output_dir)
    
    # Load problems
    problems = None
    if args.eval_file:
        eval_path = Path(args.eval_file)
        if eval_path.exists():
            problems = evaluator.load_humaneval(str(eval_path))
            console.print(f"[green]Loaded {len(problems)} problems from {args.eval_file}[/green]")
        else:
            console.print(f"[yellow]File not found: {args.eval_file}, using built-in problems[/yellow]")
    
    try:
        # Run evaluation
        if args.benchmark == "humaneval":
            results = evaluator.evaluate_humaneval(problems, max_problems=args.max_problems)
        else:
            console.print(f"[red]Unknown benchmark: {args.benchmark}[/red]")
            return
        
        # Print and save results
        evaluator.print_results(results)
        evaluator.save_results(results)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Evaluation error: {e}[/red]")
        import traceback
        traceback.print_exc()


def run_interactive(args):
    """
    Run interactive REPL mode.
    
    Args:
        args: Command-line arguments
    """
    console.print("\n[bold green]AAF Interactive Mode[/bold green]")
    console.print("Type your task description and press Enter.")
    console.print("Type 'quit' or 'exit' to stop.\n")
    
    # Initialize system once
    memory = DualLayerMemorySystem(faiss_index_path="./data/interactive_pkb")
    orchestrator = AgentOrchestrator(memory)
    
    task_num = 0
    
    while True:
        try:
            # Get task from user
            task_desc = console.input(f"[cyan]Task {task_num + 1}>[/cyan] ")
            
            if task_desc.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            if not task_desc.strip():
                continue
            
            task_num += 1
            
            # Create task
            task = TaskSpec(
                task_id=str(uuid.uuid4()),
                raw_description=task_desc,
                language="python"
            )
            
            # Run pipeline
            console.print("\n[yellow]Processing...[/yellow]\n")
            state = orchestrator.run_pipeline(task)
            
            # Print results
            console.print(f"\n[bold]Status:[/bold] {state.status}")
            console.print(f"[bold]Steps:[/bold] {state.step_count}")
            
            if "VA_approved_code" in state.agent_outputs:
                code = state.agent_outputs["VA_approved_code"]
                console.print(f"\n[bold]Generated Code:[/bold]")
                syntax = Syntax(code[:500], "python", theme="monokai")
                console.print(syntax)
                if len(code) > 500:
                    console.print("...(truncated)\n")
            else:
                console.print("\n[yellow]No code generated[/yellow]")
            
            console.print("")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            continue


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Agent Framework (AAF) — Software Engineering Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a task
  python main.py solve --task "Write a function to reverse a string"
  
  # Solve with file input
  python main.py solve --task-file task.json --output solution.py
  
  # Run evaluation
  python main.py eval --benchmark humaneval --max-problems 5
  
  # Interactive mode
  python main.py interactive
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Solve command
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a single coding task"
    )
    solve_parser.add_argument(
        "--task",
        type=str,
        help="Task description (quoted string)"
    )
    solve_parser.add_argument(
        "--task-file",
        type=str,
        help="Path to JSON task file"
    )
    solve_parser.add_argument(
        "--repo",
        type=str,
        help="Repository path (for SWE-bench style tasks)"
    )
    solve_parser.add_argument(
        "--output",
        type=str,
        help="Save solution to file"
    )
    solve_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run benchmark evaluation"
    )
    eval_parser.add_argument(
        "--benchmark",
        choices=["humaneval"],
        default="humaneval",
        help="Benchmark to run"
    )
    eval_parser.add_argument(
        "--eval-file",
        type=str,
        help="Path to benchmark JSON file"
    )
    eval_parser.add_argument(
        "--max-problems",
        type=int,
        default=10,
        help="Maximum number of problems to evaluate"
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    # Interactive command
    subparsers.add_parser(
        "interactive",
        help="Interactive REPL mode"
    )
    
    # Parse args
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Route to appropriate handler
    if args.command == "solve":
        run_single_task(args)
    elif args.command == "eval":
        run_evaluation(args)
    elif args.command == "interactive":
        run_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
