# Autonomous Agent Framework (AAF)

## Research Implementation

A complete Python implementation of the Autonomous Agent Framework for software development, based on the research paper "An Integrated Autonomous Agent Framework for Software Development" by Mistry Rohan (Swarrnim Startup and Innovation University).

## Overview

The AAF is a three-module autonomous system that generates software solutions through multi-agent coordination:

1. **Hierarchical Reasoning Engine (HRE)** - Two-tier reasoning combining ReAct with Tree-of-Thoughts
2. **Dual-Layer Adaptive Memory System (DLAMS)** - Working memory + FAISS-based persistent knowledge base
3. **Multi-Agent Coordination Protocol (MACP)** - Five specialist agents orchestrated through structured messaging

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATOR                       │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │  RA  │→ │  AA  │→ │  IA  │⇄│  VA  │→ │ INA  │            │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘           │
│     ↓          ↓         ↓         ↓         ↓              │
│  ┌─────────────────────────────────────────────────┐        │ 
│  │         DUAL-LAYER MEMORY SYSTEM (DLAMS)        │        │
│  │  ┌──────────────────┐  ┌──────────────────┐     │        │
│  │  │ Working Memory   │  │  PKB (FAISS)     │     │        │
│  │  │ (Sliding Window) │  │  (Semantic DB)   │     │        │
│  │  └──────────────────┘  └──────────────────┘     │        │
│  └─────────────────────────────────────────────────┘        │
│                         ↑                                   │
│              ┌──────────────────────┐                       │
│              │ Reasoning Engine     │                       │
│              │ (ReAct + ToT)        │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

Agents:
  RA  - Requirements Agent (task analysis)
  AA  - Architecture Agent (design)
  IA  - Implementation Agent (coding with HRE)
  VA  - Verification Agent (testing & security)
  INA - Integration Agent (deployment)
```

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Docker (optional, for sandboxed code execution)

### Setup

```bash
# Clone or download the repository
cd aaf_system

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Configuration (.env)

Create a `.env` file with the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for reasoning |
| `FAISS_INDEX_PATH` | `./data/pkb_index` | Path for FAISS index storage |
| `WORKING_MEMORY_SIZE` | `12` | Max working memory records |
| `TOT_CONFIDENCE_THRESHOLD` | `0.60` | Threshold for ToT activation |
| `TOT_BRANCHES` | `3` | Number of ToT branches to explore |
| `MAX_STEPS` | `50` | Maximum reasoning steps per task |
| `LOG_DIR` | `./logs` | Directory for audit logs |

## Quick Start

### Solve a Single Task

```bash
# Simple task
python main.py solve --task "Write a function to find the longest palindrome in a string"

# With output file
python main.py solve --task "Implement a binary search tree" --output solution.py --verbose

# From JSON file
python main.py solve --task-file task.json
```

### Run Evaluation

```bash
# Run on 10 HumanEval problems (built-in)
python main.py eval --benchmark humaneval --max-problems 10

# Use custom problem set
python main.py eval --benchmark humaneval --eval-file problems.json --max-problems 20
```

### Interactive Mode

```bash
python main.py interactive
```

This launches a REPL where you can continuously input tasks and see results.

## File Structure

```
aaf_system/
├── .env                      # Configuration (create from .env.example)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── main.py                   # CLI entry point
├── models.py                 # Core Pydantic data models
├── memory.py                 # DLAMS implementation
├── reasoning_engine.py       # HRE with ReAct + ToT
├── agents.py                 # MACP agents + orchestrator
├── evaluator.py              # HumanEval evaluation harness
│
├── data/                     # FAISS indices (created at runtime)
│   └── pkb_index.index
│   └── pkb_index.json
│
├── logs/                     # Audit logs (JSONL format)
│   └── audit_<session_id>.jsonl
│
├── workspace/                # Temp files for code execution
│   └── (temporary Python files)
│
├── results/                  # Evaluation results
│   └── eval_<timestamp>.json
│
└── tests/                    # Test suite
    └── test_aaf.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_aaf.py::test_working_memory_eviction -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Architecture Details

### 1. Hierarchical Reasoning Engine (HRE)

The HRE implements a two-tier reasoning architecture:

- **Primary Level**: ReAct (Reasoning + Acting) loop for step-by-step problem solving
- **Meta Level**: Tree-of-Thoughts for branch selection when confidence drops below threshold

**Key Features**:
- Confidence-based escalation to ToT
- Consecutive failure tracking
- Action execution sandbox (Docker-based)
- Audit logging for full traceability

### 2. Dual-Layer Adaptive Memory System (DLAMS)

DLAMS maintains both short-term and long-term memory:

- **Working Memory**: Sliding window with priority-based eviction (default: 12 records)
- **Persistent Knowledge Base**: FAISS vector store for semantic retrieval

**Key Features**:
- Priority scoring for memory importance
- Semantic search with OpenAI embeddings
- Automatic solution archiving
- Context-aware prompt injection

### 3. Multi-Agent Coordination Protocol (MACP)

Five specialist agents work together through structured messaging:

1. **Requirements Agent (RA)**: Analyzes tasks and extracts requirements
2. **Architecture Agent (AA)**: Designs software architecture
3. **Implementation Agent (IA)**: Writes code using HRE
4. **Verification Agent (VA)**: Tests and validates code
5. **Integration Agent (INA)**: Integrates into target repository

**Key Features**:
- IA↔VA revision loop (max 3 cycles)
- Structured message validation
- Shared state management
- Complete audit trail

## Research Context

This implementation is based on:

**Paper**: "An Integrated Autonomous Agent Framework for Software Development"  
**Author**: Mistry Rohan  
**Institution**: Swarrnim Startup and Innovation University  
**Date**: 2024

The paper introduces a novel three-module architecture combining hierarchical reasoning, adaptive memory, and multi-agent coordination for autonomous software development. This implementation demonstrates the feasibility and effectiveness of the proposed framework on the HumanEval benchmark.

**Companion Survey**: "A Comprehensive Survey of LLM-Based AI Agents for Software Engineering: Architectures, Applications, and Future Directions" provides broader context on LLM-based agents in software engineering.

## Performance

### Expected Performance (HumanEval)

With GPT-4o:
- **Pass@1**: 60-75% (depending on problem complexity)
- **Mean Steps**: 8-15 steps per problem
- **Mean Time**: 30-60 seconds per problem

### Cost Estimates

- **Single Task**: $0.05 - $0.20 (depending on complexity)
- **HumanEval-10**: $0.50 - $2.00
- **HumanEval-164** (full): $8 - $30

## Limitations

1. **API Dependency**: Requires OpenAI API access
2. **Docker Recommended**: For safe code execution (falls back to direct execution if unavailable)
3. **Rate Limits**: May require delays between tasks for large evaluations
4. **Context Window**: Very complex tasks may exceed context limits
5. **Cost**: Full evaluations can be expensive with large problem sets

## Troubleshooting

### FAISS Issues

```bash
# If "FAISS not found"
pip install faiss-cpu

# For GPU support (requires CUDA)
pip install faiss-gpu
```

### Docker Not Available

The system will fall back to direct Python execution (less safe). To use Docker:

```bash
# Install Docker Desktop
# Ensure Docker daemon is running
docker run hello-world  # Test installation
```

### Rate Limits

If encountering OpenAI rate limits:

1. Reduce `max_problems` in evaluation
2. Add delays between tasks (modify `evaluator.py`)
3. Use a higher-tier API key

### Embedding Dimension Mismatch

If switching embedding models, update `embedding_dim` in `memory.py`:
- `text-embedding-3-large`: 3072 dims
- `text-embedding-ada-002`: 1536 dims

## Citation

```bibtex
@article{mistry2024aaf,
  title={An Integrated Autonomous Agent Framework for Software Development},
  author={Mistry, Rohan},
  journal={Swarrnim Startup and Innovation University},
  year={2024}
}
```

## License

MIT License

Copyright (c) 2024 Mistry Rohan, Swarrnim Startup and Innovation University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This implementation was created as part of research at Swarrnim Startup and Innovation University. Special thanks to the open-source community for tools like LangChain, FAISS, and OpenAI's API.

## Contributing

This is a research implementation. For bug reports or suggestions, please open an issue or submit a pull request.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the audit logs in `./logs/`
3. Run with `--verbose` flag for detailed output
4. Consult the research paper for architectural details
