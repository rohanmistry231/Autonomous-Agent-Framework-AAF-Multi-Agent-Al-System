# AAF Quick Start Guide

This guide will get you up and running with the Autonomous Agent Framework in 5 minutes.

## Prerequisites

- Python 3.11+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Step 1: Install Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Configure API Key

Edit the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Step 3: Run Your First Task

```bash
python main.py solve --task "Write a Python function to check if a number is prime"
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║   AUTONOMOUS AGENT FRAMEWORK (AAF) v1.0                  ║
║   Research Implementation — Mistry Rohan                 ║
║   Swarrnim Startup and Innovation University             ║
╚══════════════════════════════════════════════════════════╝

Task: Write a Python function to check if a number is prime...
Task ID: ...

[RA] Analyzing requirements...
[AA] Designing architecture...
[IA] Implementing solution...
[VA] Verifying implementation...
[INA] Integrating solution...

RESULTS
Status: completed
Steps Used: 12
Final Code:
def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

## Step 4: Try More Examples

### Save output to file
```bash
python main.py solve \
  --task "Write a function to find the longest common subsequence" \
  --output lcs_solution.py \
  --verbose
```

### Interactive mode
```bash
python main.py interactive
```

### Run evaluation
```bash
python main.py eval --benchmark humaneval --max-problems 3
```

## Step 5: Understand the Output

The system creates:
- `logs/` - Audit logs (JSONL format) showing all reasoning steps
- `data/` - FAISS knowledge base (persists learned solutions)
- `results/` - Evaluation results (JSON format)

## Common Issues

### "OPENAI_API_KEY not set"
→ Edit `.env` file and add your API key

### "Docker not available"
→ Optional: Install Docker for sandboxed execution, or use without it

### "Rate limit error"
→ Wait a moment or reduce `max_problems` in evaluations

## Next Steps

1. Read `README.md` for full documentation
2. Run tests: `pytest tests/ -v`
3. Explore the architecture in the code
4. Try custom tasks and benchmarks

## Architecture Overview

```
Your Task → RA (Requirements) → AA (Architecture) → IA (Implementation)
                                                            ↓
                                                    Uses HRE (ReAct+ToT)
                                                            ↓
                                                    Stores in Memory (DLAMS)
                                                            ↓
            INA (Integration) ← VA (Verification) ← Generated Code
```

## Example Tasks to Try

1. **Simple**: "Write a function to reverse a string"
2. **Medium**: "Implement a binary search tree with insert and search"
3. **Complex**: "Create a LRU cache with O(1) operations"
4. **Real-world**: "Parse and validate JSON with error handling"

## Performance Tips

1. **Faster results**: Use `gpt-3.5-turbo` in `.env` (less accurate)
2. **Better quality**: Keep `gpt-4o` (default, recommended)
3. **Save money**: Start with small evaluations (max-problems=5)

## Support

- Check `README.md` for detailed docs
- Review audit logs in `logs/` for debugging
- Use `--verbose` flag for detailed output

---

**Ready to build something amazing? Start coding! 🚀**
