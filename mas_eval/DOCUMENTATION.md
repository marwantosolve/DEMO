# MAS Evaluation Framework Documentation v0.2.0

> **Evaluating REAL Multi-Agent Systems with Google ADK**

---

## Table of Contents

1. [Evaluating REAL Google ADK Agents](#1-evaluating-real-google-adk-agents)
2. [Metrics - Research Paper Origins](#2-metrics---research-paper-origins)
3. [MAST Classifier Explained](#3-mast-classifier-explained)
4. [Training for Domain-Specific Data](#4-training-for-domain-specific-data)

---

## 1. Evaluating REAL Google ADK Agents

> **This framework is designed to work with REAL ADK agents, NOT simulated spans.**

### Using the ADKAdapter

The `ADKAdapter` in `adapters/adk_adapter.py` automatically captures execution traces from real Google ADK agents.

```python
import asyncio
from google.adk.agents import Agent
from mas_eval.adapters import ADKAdapter
from mas_eval import MASEvaluator

# Step 1: Create your REAL ADK agents
orchestrator = Agent(
    name="Orchestrator",
    model="gemini-2.0-flash",
    description="Coordinates research tasks",
    instruction="You coordinate research between specialized agents...",
    sub_agents=[researcher, writer]  # Your other agents
)

# Step 2: Wrap with ADKAdapter
adapter = ADKAdapter(orchestrator)

# Step 3: Run with tracing - this captures REAL agent execution
async def evaluate_real_agents():
    response, spans = await adapter.run_with_tracing(
        user_message="Research AI in Healthcare"
    )
    
    # Step 4: Evaluate using captured spans
    evaluator = MASEvaluator(
        enable_crg=True,
        enable_gemmas=True,
        enable_mast=True
    )
    result = evaluator.evaluate(spans)
    
    print(result.summary())
    return result

# Run it
result = asyncio.run(evaluate_real_agents())
```

### Complete Working Example

See `examples/research_mas.py` for a complete 4-agent research MAS:

```python
from mas_eval.examples.research_mas import ResearchMAS
from mas_eval.adapters import ADKAdapter
from mas_eval import MASEvaluator
import asyncio

# Create the research MAS with REAL ADK agents
mas = ResearchMAS(model="gemini-2.0-flash")
root_agent = mas.create_agents()  # Creates Orchestrator, Academic_Researcher, Industry_Researcher, Writer

# Wrap with adapter for evaluation
adapter = ADKAdapter(root_agent)

# Run and evaluate
async def run_and_evaluate():
    response, spans = await adapter.run_with_tracing("Research renewable energy trends")
    
    # Evaluate the REAL agent execution
    evaluator = MASEvaluator()
    result = evaluator.evaluate(spans)
    
    # Generate HTML report
    evaluator.to_html(result, "my_evaluation_report.html")
    
    return result

result = asyncio.run(run_and_evaluate())
```

### What the ADKAdapter Captures

The adapter automatically captures these events from real ADK execution:

| Event Type | What's Captured |
|------------|-----------------|
| `model_turn` | Agent thoughts/reasoning |
| `function_call` | Tool invocations |
| `function_response` | Tool results |
| `agent_transfer` | Handoffs between agents |
| `text` | Final outputs |

---

## 2. Metrics - Research Paper Origins

### ⚠️ IMPORTANT: Paper Sources

| Metric | Source | Notes |
|--------|--------|-------|
| **IDS** | GEMMAS Paper | Published research |
| **UPR** | GEMMAS Paper | Published research |
| **TRS** | ❌ NO PAPER | Custom metric for this framework |
| **MAST 14 Modes** | MAST Paper | "Why Do Multi-Agent LLM Systems Fail?" |

---

### IDS & UPR - From GEMMAS Paper

**Paper**: "GEMMAS: Generalizable Multi-agent Multi-task Evaluation on Structural reasoning"

**IDS (Information Diversity Score)**:
- Measures semantic diversity of agent outputs
- Formula: `IDS = 1 - Average(Pairwise Cosine Similarity)`
- Uses sentence-transformers for embedding
- High IDS = Agents contribute unique information
- Low IDS = Agents parrot each other

**UPR (Unnecessary Path Ratio)**:
- Measures inefficiency in reasoning paths
- Formula: `UPR = 1 - (Necessary Paths / Total Paths)`
- Necessary paths = within 1.2x of shortest path length
- Low UPR = Efficient (minimal redundant steps)
- High UPR = Inefficient (many unnecessary detours)

---

### TRS (Thought Relevance Score) - CUSTOM METRIC (NOT FROM A PAPER)

> ⚠️ **TRS is NOT from a research paper.** It was created specifically for this framework.

**What it does**:
- Measures semantic similarity between agent thoughts and:
  1. The task description
  2. Final outputs
  3. Action content
- Uses cosine similarity of sentence embeddings

**How it's calculated**:
```python
# Simplified TRS calculation
thought_embedding = encoder.encode(thought_content)
reference_embeddings = encoder.encode(task_description + outputs)

max_similarity = max(cosine_similarity(thought, references))
avg_similarity = mean(cosine_similarity(thought, references))

TRS = 0.8 * max_similarity + 0.2 * avg_similarity
```

**Interpretation**:
- ≥0.7: Excellent - thoughts are highly relevant
- 0.5-0.7: Good - mostly on-topic
- 0.3-0.5: Fair - some tangents
- <0.3: Poor - largely irrelevant

---

### MAST Taxonomy - From Research Paper

**Paper**: "Why Do Multi-Agent LLM Systems Fail?" (2024)

This paper analyzed real MAS failures and identified 14 failure modes across 3 categories:

**Category 1: Specification Issues (5 modes)**
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| SPEC-1 | Disobey Task Specification | 12.5% |
| SPEC-2 | Disobey Role Specification | 8.3% |
| SPEC-3 | Step Repetition | 15.2% |
| SPEC-4 | Loss of Conversation History | 6.8% |
| SPEC-5 | Unaware of Termination Conditions | 5.4% |

**Category 2: Inter-Agent Misalignment (6 modes)**
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| INTER-1 | Conversation Reset | 2.33% |
| INTER-2 | Fail to Ask for Clarification | 11.65% |
| INTER-3 | Task Derailment | 7.15% |
| INTER-4 | Information Withholding | 1.66% |
| INTER-5 | Ignored Other Agent's Output | 0.17% |
| INTER-6 | Reasoning-Action Mismatch | 13.98% |

**Category 3: Task Verification (3 modes)**
| Code | Failure Mode | Frequency |
|------|-------------|-----------|
| TASK-1 | Premature Termination | 8.7% |
| TASK-2 | Fail to Identify Task Completion | 4.2% |
| TASK-3 | Fail to Identify Task Failure | 3.8% |

---

## 3. MAST Classifier Explained

### Is it Fine-Tuned on Real MAST Data?

**NO.** The classifier uses **prompt engineering**, not actual training.

```
┌─────────────────────────────────────────────┐
│ How MAST Classification Works               │
├─────────────────────────────────────────────┤
│ 1. Format trace data as text                │
│ 2. Build prompt with 14 failure definitions │
│ 3. Send to Gemini (general-purpose LLM)     │
│ 4. Parse JSON response                      │
│ 5. Validate against known failure codes     │
└─────────────────────────────────────────────┘
```

### Classification Modes

| Mode | What It Really Is |
|------|-------------------|
| `ZERO_SHOT` | Basic prompt with taxonomy |
| `FEW_SHOT` | Prompt with 3 example classifications |
| `FINE_TUNED` | **Enhanced chain-of-thought prompting** (NOT actual fine-tuning!) |

---

## 4. Training for Domain-Specific Data

### For Telecom or Other Special Domains

**Quick Approach (No Training)**:
1. Add domain context to prompts
2. Use few-shot examples with telecom-specific failures

**Production Approach**:
1. Collect 500+ labeled telecom MAS traces
2. Fine-tune using Vertex AI
3. Use your fine-tuned model:

```python
classifier = MASTClassifier(
    model="projects/PROJECT/locations/LOCATION/endpoints/YOUR_MODEL",
    mode=ClassifierMode.FINE_TUNED
)
```

---

## 5. Architecture Reference

```
mas_eval/
├── evaluator.py         # MASEvaluator - unified API
├── adapters/
│   └── adk_adapter.py   # ADKAdapter for REAL ADK agents
├── examples/
│   └── research_mas.py  # Complete 4-agent example
├── graph/
│   └── crg_builder.py   # Causal Reasoning Graph
├── metrics/
│   ├── gemmas.py        # IDS/UPR (from GEMMAS paper)
│   └── thought_relevance.py  # TRS (custom, no paper)
├── mast/
│   ├── taxonomy.py      # 14 failure modes (from MAST paper)
│   └── classifier.py    # LLM-based classification
└── suggestions/
    └── advisor.py       # Improvement suggestions
```

---

### Summary

1. **Use `ADKAdapter`** to evaluate real Google ADK agents
2. **IDS/UPR** come from the GEMMAS research paper
3. **TRS is NOT from a paper** - it's a custom metric
4. **MAST taxonomy** comes from "Why Do Multi-Agent LLM Systems Fail?" paper
5. **The classifier is NOT trained** - it uses prompt engineering with Gemini
