# AAF System Architecture

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (CLI)                         │
│                         main.py                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                  │
│  │  solve   │  │   eval   │  │ interactive  │                  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘                  │
└───────┼─────────────┼────────────────┼──────────────────────────┘
        │             │                │
        ▼             ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│              AGENT ORCHESTRATOR (MACP)                           │
│                      agents.py                                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │
│  │  RA  │→→│  AA  │→→│  IA  │⇄⇄│  VA  │→→│ INA  │             │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘             │
│     │         │         │         │         │                    │
│     └─────────┴─────────┴─────────┴─────────┘                    │
│                      │                                            │
│                      ▼                                            │
│  ┌────────────────────────────────────────────────┐             │
│  │   HIERARCHICAL REASONING ENGINE (HRE)          │             │
│  │          reasoning_engine.py                   │             │
│  │  ┌──────────────┐  ┌──────────────────┐       │             │
│  │  │ ReAct Loop   │  │ Tree-of-Thoughts │       │             │
│  │  │ (Primary)    │  │  (Meta-level)    │       │             │
│  │  └──────────────┘  └──────────────────┘       │             │
│  └────────────────────────────────────────────────┘             │
│                      │                                            │
│                      ▼                                            │
│  ┌────────────────────────────────────────────────┐             │
│  │  DUAL-LAYER ADAPTIVE MEMORY SYSTEM (DLAMS)    │             │
│  │               memory.py                        │             │
│  │  ┌──────────────────┐  ┌──────────────────┐  │             │
│  │  │ Working Memory   │  │  PKB (FAISS)     │  │             │
│  │  │ • 12 slots       │  │  • Semantic DB   │  │             │
│  │  │ • Priority evict │  │  • Embeddings    │  │             │
│  │  └──────────────────┘  └──────────────────┘  │             │
│  └────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DATA MODELS (models.py)                       │
│  ReasoningStep │ WorkingMemory │ PKBEntry │ TaskSpec │ ...      │
└──────────────────────────────────────────────────────────────────┘
```

## Multi-Agent Coordination Flow

```
Task Input
    │
    ▼
┌─────────────────┐
│ Requirements    │  Extracts: requirements, acceptance_criteria,
│ Agent (RA)      │            edge_cases, constraints
└────────┬────────┘
         │ task_spec message
         ▼
┌─────────────────┐
│ Architecture    │  Designs:  components, interfaces, data_flow,
│ Agent (AA)      │            design_patterns, file_structure
└────────┬────────┘
         │ arch_plan message
         ▼
┌─────────────────┐
│ Implementation  │  Generates: Complete code implementation
│ Agent (IA)      │  Uses: HRE (ReAct + ToT)
│   [Uses HRE]    │  Memory: DLAMS for context
└────────┬────────┘
         │ code_artifact message
         ▼
┌─────────────────┐
│ Verification    │  Checks: Logic, security (Bandit), tests
│ Agent (VA)      │  Returns: APPROVE | REVISE | REJECT
└────────┬────────┘
         │
         ├─→ REVISE (max 3 cycles)
         │   ↓
         │   │ correction_request message
         │   └─→ (back to IA)
         │
         └─→ APPROVE
             │ validation_result message
             ▼
┌─────────────────┐
│ Integration     │  Integrates: Code into repository
│ Agent (INA)     │  Generates: Diff, test results
└────────┬────────┘
         │ patch message
         ▼
    Final Solution
```

## Hierarchical Reasoning Engine (HRE) Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    HRE Main Loop                            │
│                                                             │
│  1. Build Memory Context                                   │
│     ├─ Working Memory (recent steps)                       │
│     └─ PKB Retrieval (semantic search)                     │
│                                                             │
│  2. Generate ReAct Step                                    │
│     ├─ LLM Call with system prompt                         │
│     └─ Parse: thought, action, action_input, confidence    │
│                                                             │
│  3. Confidence Check                                       │
│     ├─ IF confidence < 0.60 OR failures >= 2:              │
│     │   ├─ Generate N alternative branches (ToT)           │
│     │   ├─ Meta-reasoning LLM selects best branch          │
│     │   └─ Use selected branch                             │
│     └─ ELSE: Use primary ReAct step                        │
│                                                             │
│  4. Execute Action                                         │
│     ├─ write_code    → Store code                          │
│     ├─ execute_code  → Docker sandbox                      │
│     ├─ read_file     → File I/O                            │
│     ├─ search_pkb    → FAISS query                         │
│     ├─ analyze_error → Error analysis                      │
│     └─ finalize      → Complete solution                   │
│                                                             │
│  5. Update State                                           │
│     ├─ Add to working memory                               │
│     ├─ Log to audit trail                                  │
│     └─ Track failures                                      │
│                                                             │
│  6. Check Termination                                      │
│     ├─ Solution finalized? → DONE                          │
│     └─ Step limit reached? → FAILED                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Dual-Layer Memory System (DLAMS)

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKING MEMORY                           │
│                   (Short-term, 12 slots)                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Step 1: [thought, action, observation, conf=0.9] ◄──┐ │ │
│  │ Step 2: [thought, action, observation, conf=0.7]    │ │ │
│  │ Step 3: [thought, action, observation, conf=0.8]    │ │ │
│  │ ...                                                  │ │ │
│  │ Step 12: [thought, action, observation, conf=0.95]  │ │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │                                   │
│     Priority-based      │   When full, evict                │
│     eviction (0.0-1.0)  │   lowest priority                 │
│                         │                                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              PROMPT INJECTION                         │ │
│  │  "Recent steps show: Step 10 tried X, Step 11 ..."   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ Semantic Query
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           PERSISTENT KNOWLEDGE BASE (PKB)                   │
│                (Long-term, FAISS vectors)                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Entry 1: [code_solution, embedding[3072], task_hash] │ │
│  │ Entry 2: [api_doc, embedding[3072], task_hash]       │ │
│  │ Entry 3: [error_pattern, embedding[3072], task_hash] │ │
│  │ Entry N: [constraint, embedding[3072], task_hash]    │ │
│  └───────────────────────────────────────────────────────┘ │
│                         │                                   │
│  OpenAI text-embedding-3-large (3072 dimensions)           │
│  Cosine similarity via FAISS IndexFlatIP                   │
│                         │                                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           TOP-K RETRIEVAL (k=3)                       │ │
│  │  Returns most semantically relevant past solutions    │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Message Flow Between Agents

```
┌──────────────────────────────────────────────────────────────┐
│                    AgentMessage Schema                       │
│  {                                                           │
│    message_id: UUID                                          │
│    sender: "RA" | "AA" | "IA" | "VA" | "INA"                │
│    recipient: "RA" | "AA" | "IA" | "VA" | "INA"             │
│    message_type: "task_spec" | "arch_plan" | ...            │
│    payload: {...}  ← Type-specific data                     │
│    timestamp: datetime                                       │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Payload Validation (PAYLOAD_SCHEMAS)            │
│                                                              │
│  task_spec:        [requirements, acceptance_criteria, ...]  │
│  arch_plan:        [components, interfaces, data_flow]       │
│  code_artifact:    [code, language, file_path, test_spec]    │
│  error_report:     [errors, line_numbers, severity, ...]     │
│  patch:            [files_modified, diff, test_results]      │
│  validation_result:[passed, failed_tests, coverage, ...]     │
│  correction_req:   [original_id, errors, correction_hint]    │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 SharedAgentState                             │
│  • Current task (TaskSpec)                                   │
│  • Working memory (WorkingMemoryRecord)                      │
│  • Agent outputs (dict)                                      │
│  • Message history (list[AgentMessage])                      │
│  • Step count / max steps                                    │
│  • Status (pending|running|completed|failed)                 │
│  • Audit log path                                            │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow: Task to Solution

```
User Input
    │
    ▼
┌─────────────────┐
│   TaskSpec      │  • raw_description
│                 │  • language
│                 │  • acceptance_criteria (empty initially)
│                 │  • edge_cases (empty initially)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RA Process    │  Enriches TaskSpec:
│                 │  • acceptance_criteria ← filled
│                 │  • edge_cases ← filled
│                 │  • constraints ← filled
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AA Process    │  Creates architecture:
│                 │  • components list
│                 │  • interfaces graph
│                 │  • file_structure
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   IA Process    │  Iterative coding:
│   [HRE Loop]    │  Step 1: Analyze task
│                 │  Step 2: Search PKB for similar solutions
│                 │  Step 3: Write initial code
│                 │  Step 4: Execute and test
│                 │  Step 5: Analyze errors (if any)
│                 │  Step N: Finalize solution
│                 │  
│                 │  Each step:
│                 │  • Stored in WorkingMemory
│                 │  • Logged to audit trail
│                 │  • Considered for PKB storage
└────────┬────────┘
         │ Generated Code
         ▼
┌─────────────────┐
│   VA Process    │  Verification:
│                 │  • Bandit security scan
│                 │  • Syntax check (compile)
│                 │  • Logic analysis (LLM)
│                 │  • Test execution
│                 │  
│                 │  Decision:
│                 │  • APPROVE → INA
│                 │  • REVISE → IA (with hints)
│                 │  • REJECT → fail
└────────┬────────┘
         │ Approved Code
         ▼
┌─────────────────┐
│  INA Process    │  Integration:
│                 │  • Write to file
│                 │  • Generate diff
│                 │  • Update imports
│                 │  • Final test run
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Solution  │  • solution.py (code file)
│                 │  • audit_log.jsonl (full trace)
│                 │  • PKB entry (for future)
└─────────────────┘
```

## Tree-of-Thoughts (ToT) Escalation

```
                    ReAct Step Generated
                            │
                            ▼
                ┌─────────────────────┐
                │ Confidence Check    │
                │ Is conf < 0.60?     │
                │ OR failures >= 2?   │
                └──────────┬──────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼ NO                            ▼ YES
    Use Primary Step              Activate ToT Meta-Reasoning
    (Fast path)                               │
                                              ▼
                              ┌─────────────────────────────┐
                              │ Generate K Branches         │
                              │ (default K=3)               │
                              │                             │
                              │ Branch 1: [temp=0.7, LLM]   │
                              │ Branch 2: [temp=0.7, LLM]   │
                              │ Branch 3: [temp=0.7, LLM]   │
                              └────────────┬────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │ Meta-Reasoning LLM          │
                              │ Evaluates all branches on:  │
                              │ • Correctness likelihood    │
                              │ • Safety                    │
                              │ • Efficiency                │
                              │ • Specificity               │
                              └────────────┬────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │ Select Best Branch          │
                              │ Returns: (branch, conf)     │
                              └────────────┬────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │ Reset Failure Counter       │
                              │ failures = 0                │
                              └─────────────────────────────┘
```

## Code Execution Sandbox

```
┌──────────────────────────────────────────────────────────────┐
│             Docker Sandbox Execution                         │
│                                                              │
│  1. Write code to temp file                                 │
│     /workspace/tmp_exec_<timestamp>.py                      │
│                                                              │
│  2. Docker Run Command:                                     │
│     docker run --rm \                                       │
│       --network=none \         ← No network access          │
│       --memory=256m \          ← Limited memory             │
│       --cpus=0.5 \             ← Limited CPU                │
│       -v ./workspace:/workspace \                           │
│       python:3.11-slim \                                    │
│       python /workspace/tmp_exec_<timestamp>.py             │
│                                                              │
│  3. Capture Output                                          │
│     ├─ stdout (normal output)                               │
│     └─ stderr (error messages)                              │
│                                                              │
│  4. Timeout Handling                                        │
│     ├─ Default: 30 seconds                                  │
│     └─ TimeoutExpired → return error                        │
│                                                              │
│  5. Cleanup                                                 │
│     └─ Delete temp file                                     │
│                                                              │
│  Fallback (if Docker unavailable):                          │
│     subprocess.run(["python", temp_file])                   │
│     (Less safe, but functional)                             │
└──────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                    Complexity Tiers                         │
│                                                             │
│  SIMPLE (1-5 steps, 10-20s)                                │
│  • "Reverse a string"                                       │
│  • "Check if number is even"                               │
│  • Cost: ~$0.02-0.05                                       │
│                                                             │
│  MEDIUM (5-15 steps, 30-60s)                               │
│  • "Implement binary search"                               │
│  • "Parse JSON with validation"                            │
│  • Cost: ~$0.05-0.15                                       │
│                                                             │
│  COMPLEX (15-30 steps, 60-120s)                            │
│  • "Build LRU cache with O(1) ops"                         │
│  • "Implement Dijkstra's algorithm"                        │
│  • Cost: ~$0.15-0.30                                       │
│                                                             │
│  VERY COMPLEX (30-50 steps, 120-300s)                      │
│  • "Design and implement REST API"                         │
│  • "Create multi-threaded queue system"                    │
│  • Cost: ~$0.30-0.50                                       │
└─────────────────────────────────────────────────────────────┘
```

---

**This architecture supports**:
- ✅ Autonomous code generation
- ✅ Multi-step reasoning with memory
- ✅ Quality assurance through verification
- ✅ Learning from past solutions
- ✅ Safe code execution
- ✅ Complete audit trails
