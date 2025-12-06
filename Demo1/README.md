# MAS Evaluation Framework Documentation v0.2.0

> **A modular, plug-and-play framework for evaluating Multi-Agent Systems (MAS)**

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Architecture & Modules](#2-architecture--modules)
3. [How Everything Works](#3-how-everything-works)
4. [MAST Classifier Deep Dive](#4-mast-classifier-deep-dive)
5. [Training & Fine-Tuning for Domain-Specific Data](#5-training--fine-tuning-for-domain-specific-data)
6. [Quick Start Guide](#6-quick-start-guide)
7. [API Reference](#7-api-reference)

---

## 1. Framework Overview

This framework provides **comprehensive evaluation capabilities** for Multi-Agent Systems by combining:

| Component | Purpose |
|-----------|---------|
| **OpenTelemetry Tracing** | Captures agent execution traces |
| **Causal Reasoning Graph (CRG)** | Models information flow between agents |
| **GEMMAS Metrics** | IDS (Information Diversity) & UPR (Unnecessary Path Ratio) |
| **Thought Relevance Score (TRS)** | Measures usefulness of agent reasoning |
| **MAST Classifier** | Detects 14 failure modes using LLM |
| **MAS Advisor** | Generates improvement suggestions |

### Key Features

- 🔌 **Plug-and-Play** - Works with any MAS framework
- 🎯 **14 Failure Modes** - Complete MAST taxonomy coverage
- 🧠 **LLM-Powered** - Uses Gemini for intelligent classification
- 📊 **Comprehensive Metrics** - Multiple evaluation dimensions
- 💡 **Actionable Insights** - Automatic improvement suggestions

---

## 2. Architecture & Modules

```
mas_eval/
├── __init__.py          # Package exports
├── evaluator.py         # MASEvaluator - Unified API
├── core/
│   ├── base.py          # Base classes (BaseModule, BaseMetric, BaseClassifier)
│   └── types.py         # Data types (Span, Node, Edge, FailureMode, etc.)
├── tracer/
│   ├── otel_tracer.py   # OpenTelemetry integration
│   └── span_collector.py # Span collection utilities
├── graph/
│   ├── crg_builder.py   # Causal Reasoning Graph builder
│   └── visualizer.py    # Graph visualization
├── metrics/
│   ├── gemmas.py        # IDS/UPR metrics from GEMMAS paper
│   ├── thought_relevance.py # TRS metric
│   └── base_metric.py   # Custom metric base class
├── mast/
│   ├── taxonomy.py      # 14 failure mode definitions
│   └── classifier.py    # LLM-based classification
├── suggestions/
│   └── advisor.py       # Improvement suggestions generator
└── adapters/            # Framework adapters (ADK, LangChain, etc.)
```

### Module Responsibilities

#### 1. **TracerModule** (`tracer/otel_tracer.py`)
- Captures agent execution as spans using OpenTelemetry
- Context managers for tracing tasks and steps
- Collects hierarchical parent-child relationships

#### 2. **CRGModule** (`graph/crg_builder.py`)
- Builds a NetworkX DiGraph from spans
- Nodes = Agent states (thoughts, actions, outputs)
- Edges = Causal dependencies (parent→child)
- Provides graph analysis methods (critical path, statistics)

#### 3. **MetricsModule** (`metrics/`)
- **IDS (Information Diversity Score)**: Measures semantic diversity of information shared
- **UPR (Unnecessary Path Ratio)**: Detects redundant reasoning paths
- **TRS (Thought Relevance Score)**: Measures how relevant agent thoughts are to the task

#### 4. **MASTClassifier** (`mast/classifier.py`)
- LLM-based failure mode detection
- Supports 14 failure modes from MAST taxonomy
- Three modes: Zero-shot, Few-shot, Fine-tuned (chain-of-thought)

#### 5. **MASAdvisor** (`suggestions/advisor.py`)
- Analyzes evaluation results
- Generates prioritized improvement suggestions
- Provides actionable advice for each detected issue

---

## 3. How Everything Works

### Evaluation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Spans     │───►│  CRG Build  │───►│   Metrics   │───►│    MAST     │
│  (Traces)   │    │   (Graph)   │    │  (IDS/UPR)  │    │ (Failures)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                    ┌─────────────────────┐
                                                    │  MAS Advisor        │
                                                    │  (Suggestions)      │
                                                    └─────────────────────┘
```

### Step-by-Step Process

1. **Input**: List of `Span` objects representing agent execution
2. **CRG Construction**: Build graph showing information flow
3. **Metric Calculation**: 
   - IDS: Semantic embedding comparison of agent outputs
   - UPR: Path analysis for redundancy detection
   - TRS: Thought-to-task relevance scoring
4. **MAST Classification**: LLM analyzes trace for failure patterns
5. **Suggestion Generation**: Based on failures and metrics, generate advice
6. **Report**: Unified `EvaluationResult` with all findings

### Data Flow

```python
# Simplified internal flow
spans: List[Span]          # Raw trace data
    ↓
graph: nx.DiGraph          # CRGModule.build(spans)
    ↓
metrics: Dict[str, float]  # MetricsModule.evaluate(graph, spans)
    ↓
failures: List[FailureMode] # MASTClassifier.classify(spans)
    ↓
suggestions: List[Suggestion] # MASAdvisor.generate_suggestions(result)
```

---

## 4. MAST Classifier Deep Dive

### ⚠️ Important: The Classifier is NOT Fine-Tuned on Real MAST Dataset

> **The MAST "classifier" in this framework is NOT a traditionally trained or fine-tuned model.** 

It uses **prompt engineering** with a general-purpose LLM (Gemini) to perform classification. Here's the critical distinction:

### How It Actually Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                       MAST Classification Flow                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   1. PROMPT CONSTRUCTION                                              │
│      ├─► Taxonomy context (14 failure modes with descriptions)       │
│      ├─► Chain-of-thought reasoning instructions                     │
│      └─► Trace data (formatted spans)                                │
│                                                                       │
│   2. LLM INFERENCE (Gemini 2.0 Flash)                                │
│      ├─► General-purpose LLM (not specialized for MAST)              │
│      ├─► Uses in-context learning from prompt                        │
│      └─► Outputs JSON with detected failures                         │
│                                                                       │
│   3. POST-PROCESSING                                                  │
│      ├─► Parse JSON response                                         │
│      ├─► Validate against taxonomy codes                             │
│      ├─► Calibrate confidence scores                                 │
│      └─► Return FailureMode objects                                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Classification Modes

| Mode | Description | Implementation |
|------|-------------|----------------|
| **ZERO_SHOT** | Basic prompt with taxonomy | Single prompt, no examples |
| **FEW_SHOT** | Includes 3 example classifications | Prompt with examples |
| **FINE_TUNED** | Enhanced chain-of-thought reasoning | Detailed analysis prompts |

> **Note**: "FINE_TUNED" mode is **misleading** - it's actually enhanced prompting, NOT an actual fine-tuned model.

### The "Fine-Tuned" Prompt Strategy

The `FINE_TUNED` mode uses chain-of-thought (CoT) prompting with:

1. **Detailed Taxonomy Reference**: All 14 modes with indicators and severity
2. **Analysis Methodology**: 5-step structured analysis
3. **Specific Output Format**: JSON with confidence calibration
4. **Evidence Requirements**: Must cite specific trace content

```python
# From classifier.py - The "fine-tuned" prompt
def _fine_tuned_prompt(self, trace_text: str, taxonomy: str) -> str:
    """Build fine-tuned classification prompt with chain-of-thought reasoning."""
    return f"""You are an expert Multi-Agent System (MAS) failure analyst...
    
    ## ANALYSIS METHODOLOGY
    Follow this chain-of-thought analysis process:
    
    ### Step 1: Trace Overview
    ### Step 2: Per-Agent Analysis
    ### Step 3: Inter-Agent Communication Analysis
    ### Step 4: Task Completion Analysis
    ### Step 5: Pattern Matching to Failure Modes
    ...
    """
```

---

## 5. Training & Fine-Tuning for Domain-Specific Data

### Do You Need to Train/Fine-Tune for Telecom Data?

**Short Answer**: It depends on your use case and accuracy requirements.

### Detailed Analysis

#### Current Approach (No Training Required)

The current implementation uses **prompt engineering** with Gemini, which means:

✅ **Pros**:
- Works out-of-the-box for any domain
- No training data required
- Leverages Gemini's general knowledge
- Easy to modify prompts for domain context

❌ **Cons**:
- May miss domain-specific failure patterns
- Generic understanding of telecom terminology
- No learned patterns from real telecom failures
- Confidence scores may not reflect domain reality

#### When You SHOULD Consider Fine-Tuning

| Scenario | Recommendation |
|----------|----------------|
| **Validating a research prototype** | ❌ Current approach is fine |
| **Production telecom monitoring** | ✅ Fine-tuning recommended |
| **Detecting rare telecom-specific failures** | ✅ Fine-tuning required |
| **Compliance/regulatory requirements** | ✅ Fine-tuning + validation |
| **Generic MAS evaluation research** | ❌ Current approach works |

### Options for Domain Adaptation

#### Option 1: Enhanced Prompt Engineering (No Training)

Add telecom-specific context to prompts:

```python
# Example: Adding telecom context
classifier = MASTClassifier(model="gemini-2.0-flash")

# Modify the taxonomy context with telecom-specific indicators
telecom_context = """
DOMAIN: Telecom Network Management
SPECIFIC FAILURE PATTERNS TO WATCH:
- Timeout failures in API calls
- Provisioning sequence errors
- SLA violation cascades
- Resource allocation conflicts
- Configuration drift between agents
"""

# You would need to modify classifier._build_prompt() 
# to include this domain context
```

#### Option 2: Few-Shot Learning with Domain Examples

Create telecom-specific examples:

```python
# Add telecom examples to the few-shot prompt
telecom_examples = """
EXAMPLE: Telecom Provisioning Failure
Trace: Agent-A: "Starting 5G provisioning". Agent-B: "Checking spectrum allocation". 
       Agent-C: "User activated on 4G instead of 5G"
Output: {"failures": [{"code": "INTER-3", "confidence": 0.9, 
         "evidence": "Agent-C activated wrong network type despite 5G request"}]}

EXAMPLE: Telecom Configuration Drift  
Trace: Agent-A: "Config version 2.1". Agent-B: "Applied config 1.9". 
       Agent-C: "Validated version 2.1"
Output: {"failures": [{"code": "INTER-5", "confidence": 0.85,
         "evidence": "Agent-B ignored Agent-A's config version"}]}
"""
```

#### Option 3: True Fine-Tuning on Telecom Data

For production systems, consider:

1. **Collect Labeled Data**: 
   - Gather real telecom MAS traces
   - Manually label failure modes
   - Minimum 100-500 examples per failure type

2. **Fine-Tune with Vertex AI**:
   ```python
   # Using Google Cloud Vertex AI for fine-tuning
   from google.cloud import aiplatform
   
   aiplatform.init(project="your-project", location="us-central1")
   
   # Create tuning job with your telecom dataset
   tuning_job = aiplatform.TuningJob.create(
       training_data="gs://bucket/telecom_failures.jsonl",
       base_model="gemini-1.5-flash-002",
       tuned_model_display_name="telecom-mast-classifier"
   )
   ```

3. **Integrate Fine-Tuned Model**:
   ```python
   # Use your fine-tuned model
   classifier = MASTClassifier(
       model="projects/PROJECT/locations/LOCATION/endpoints/TUNED_MODEL",
       mode=ClassifierMode.FINE_TUNED
   )
   ```

#### Option 4: Add Telecom-Specific Failure Modes

Extend the taxonomy with telecom patterns:

```python
from mas_eval.mast.taxonomy import FailureModeDefinition, FailureCategory

# Add telecom-specific failure modes
TELECOM_1 = FailureModeDefinition(
    code="TELECOM-1",
    name="SLA Violation Cascade",
    category=FailureCategory.SPECIFICATION,
    description="Agent fails to maintain SLA requirements, causing downstream violations",
    indicators=[
        "Timeout exceeded in provisioning",
        "Latency requirements not met",
        "Throughput below SLA threshold"
    ],
    severity="high",
    frequency=15.0
)

# Register with taxonomy
taxonomy = MASTTaxonomy()
taxonomy._modes["TELECOM-1"] = TELECOM_1
```

### Recommended Approach for Telecom

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Telecom Domain Adaptation Roadmap                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Quick Validation (No Training)                                │
│  ├─► Use current framework with enhanced prompts                        │
│  ├─► Add telecom examples to few-shot mode                             │
│  └─► Evaluate accuracy on sample traces                                 │
│                                                                          │
│  Phase 2: Domain Specialization (If Needed)                             │
│  ├─► Collect 100+ labeled telecom failure traces                       │
│  ├─► Add telecom-specific failure modes to taxonomy                    │
│  └─► Implement custom confidence calibration                           │
│                                                                          │
│  Phase 3: Production Fine-Tuning (For Critical Systems)                 │
│  ├─► Prepare training dataset (500+ examples)                          │
│  ├─► Fine-tune using Vertex AI or similar                              │
│  └─► A/B test against prompt-based approach                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Quick Start Guide

### Basic Usage

```python
from mas_eval import MASEvaluator
from mas_eval.core.types import Span, StepType
from datetime import datetime

# Create sample spans
spans = [
    Span(
        span_id="1",
        name="Orchestrator.start",
        agent_name="Orchestrator",
        step_type=StepType.ACTION,
        content="Starting telecom provisioning task",
        start_time=datetime.now()
    ),
    Span(
        span_id="2",
        parent_span_id="1",
        name="NetworkAgent.thought",
        agent_name="NetworkAgent",
        step_type=StepType.THOUGHT,
        content="I need to check available spectrum...",
        start_time=datetime.now()
    ),
    # ... more spans
]

# Evaluate
evaluator = MASEvaluator(
    enable_crg=True,
    enable_gemmas=True,
    enable_mast=True  # Requires GOOGLE_API_KEY
)

result = evaluator.evaluate(spans)
print(result.summary())
```

### With Custom Task Description

```python
# For better TRS calculation
from mas_eval.metrics import ThoughtRelevanceMetric

trs = ThoughtRelevanceMetric()
trs_result = trs.calculate(
    spans=spans,
    task_description="Provision 5G network slice for enterprise customer"
)
print(f"Thought Relevance: {trs_result['overall_score']}")
```

---

## 7. API Reference

### MASEvaluator

```python
MASEvaluator(
    enable_tracing: bool = True,     # OpenTelemetry tracing
    enable_crg: bool = True,         # CRG building
    enable_gemmas: bool = True,      # IDS/UPR metrics
    enable_mast: bool = True,        # MAST failure detection
    mast_model: str = "gemini-2.0-flash",
    mast_confidence: float = 0.5,    # Min confidence threshold
    custom_metrics: Dict = None
)

# Methods
.evaluate(spans) -> EvaluationResult
.to_html(result, path) -> None
.quick_summary(result) -> str
```

### MASTClassifier

```python
MASTClassifier(
    model: str = "gemini-2.0-flash",
    mode: ClassifierMode = ClassifierMode.ZERO_SHOT,
    confidence_threshold: float = 0.5,
    api_key: str = None  # Uses GOOGLE_API_KEY env var
)

# Methods
.classify(spans) -> ClassificationResult
.summary(result) -> str
.get_taxonomy() -> MASTTaxonomy
```

### ThoughtRelevanceMetric

```python
ThoughtRelevanceMetric(
    model_name: str = "all-MiniLM-L6-v2"  # Sentence transformer
)

# Methods
.calculate(graph, spans, task_description) -> Dict
.get_improvement_suggestions(result) -> List[str]
```

---

## Summary: Key Takeaways

1. **The MAST classifier is NOT trained on real data** - It uses prompt engineering with Gemini
2. **For telecom data**, you have options:
   - Quick: Enhanced prompts + few-shot examples
   - Medium: Custom failure modes + confidence calibration
   - Full: Fine-tuning on labeled telecom traces
3. **The "Fine-Tuned" mode is actually chain-of-thought prompting**, not actual model fine-tuning
4. **For production systems**, real fine-tuning on domain data is recommended

---

*Documentation generated for MAS Evaluation Framework v0.2.0*
