"""
MAST Classifier - LLM-based failure mode detection.

Uses LLMs (Gemini or fine-tuned models) to classify MAS traces
into MAST failure modes.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from mas_eval.core.base import BaseClassifier
from mas_eval.core.types import Span, FailureMode, ConfidenceBreakdown
from mas_eval.mast.taxonomy import MASTTaxonomy, FailureCategory


class ClassifierMode(Enum):
    """Classification strategy."""
    ZERO_SHOT = "zero_shot"      # Base model + Instructions
    FEW_SHOT = "few_shot"        # Base model + Generic examples
    FEW_SHOT_ICL = "few_shot_icl" # Base model + Real MAD dataset examples
    FINE_TUNED = "fine_tuned"    # Actual fine-tuned model (minimal prompting)


@dataclass
class ClassificationResult:
    """Result of MAST classification."""
    failure_modes: List[FailureMode]
    trace_summary: str
    confidence_threshold: float
    model_used: str
    
    def to_dict(self) -> Dict:
        return {
            "failure_modes": [
                {
                    "code": f.code,
                    "name": f.name,
                    "category": f.category,
                    "confidence": f.confidence,
                    "evidence": f.evidence
                }
                for f in self.failure_modes
            ],
            "trace_summary": self.trace_summary,
            "model_used": self.model_used
        }
    
    def has_failures(self) -> bool:
        return len(self.failure_modes) > 0
    
    def get_by_category(self, category: str) -> List[FailureMode]:
        return [f for f in self.failure_modes if f.category == category]


class MASTClassifier(BaseClassifier):
    """
    LLM-based MAST failure mode classifier.
    
    Analyzes MAS traces and detects failure modes from the MAST taxonomy.
    
    Usage:
        classifier = MASTClassifier(model="gemini-2.0-flash")
        
        # Classify a trace
        result = classifier.classify(spans)
        
        for failure in result.failure_modes:
            print(f"[{failure.code}] {failure.name}: {failure.confidence:.2f}")
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        mode: ClassifierMode = ClassifierMode.ZERO_SHOT,
        confidence_threshold: float = 0.5,
        api_key: Optional[str] = None,
        dataset_path: Optional[str] = None
    ):
        """
        Initialize MAST classifier.
        
        Args:
            model: Model name (Gemini model or path to fine-tuned model)
            mode: Classification strategy
            confidence_threshold: Minimum confidence to report failure
            api_key: Gemini API key
            dataset_path: Path to MAD dataset (for FEW_SHOT_ICL mode)
        """
        super().__init__(name="MASTClassifier", model_name=model)
        
        self.model_name = model
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.dataset_path = dataset_path
        self.taxonomy = MASTTaxonomy()
        self._client = None
        self._is_tuned_model = model.startswith("tunedModels/")
        
        # Check for FINE_TUNED mode usage
        if self.mode == ClassifierMode.FINE_TUNED:
            if self._is_tuned_model:
                print(f"✅ Using fine-tuned model: {model}")
            else:
                print("⚠️  FINE_TUNED mode selected but model is not a tuned model.")
                print("   → To fine-tune, use MASTFineTuner:")
                print("   from mas_eval.mast.fine_tuning import MASTFineTuner")
                print("   tuner = MASTFineTuner()")
                print('   model_id = tuner.train_on_mad_dataset("mast_dataset/MAD_human_labelled_dataset.json")')
                print("   → Then use: MASTClassifier(model=model_id, mode=ClassifierMode.FINE_TUNED)")
        
        # Load few-shot loader only if NOT using a tuned model (tuned models don't need examples)
        self._fewshot_loader = None
        if self.mode == ClassifierMode.FEW_SHOT_ICL:
            from mas_eval.mast.mast_fewshot import MASTFewShotLoader
            self._fewshot_loader = MASTFewShotLoader(dataset_path)
            if not self._fewshot_loader.load():
                print("Warning: Failed to load MAD dataset. Fallback to FEW_SHOT.")
                self.mode = ClassifierMode.FEW_SHOT
    
    def _initialize_client(self):
        """Lazy initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                
                api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Google API key required. Set GOOGLE_API_KEY environment variable "
                        "or pass api_key to constructor."
                    )
                
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai required. Install with: pip install google-generativeai"
                )
    
    def classify(self, spans: List[Span]) -> ClassificationResult:
        """
        Classify failure modes in the trace.
        
        Args:
            spans: List of spans to analyze
        
        Returns:
            ClassificationResult with detected failures
        """
        self._initialize_client()
        
        # Prepare trace text
        trace_text = self._format_trace(spans)
        
        # Build prompt
        prompt = self._build_prompt(trace_text)
        
        # Get classification
        try:
            response = self._client.generate_content(prompt)
            result = self._parse_response(response.text, spans)
            return result
        except Exception as e:
            # Return empty result on error
            return ClassificationResult(
                failure_modes=[],
                trace_summary=f"Classification error: {str(e)}",
                confidence_threshold=self.confidence_threshold,
                model_used=self.model_name
            )
    
    def _format_trace(self, spans: List[Span]) -> str:
        """
        Format spans into analyzable text with preserved structure.
        
        Preserves:
        - Parent-child relationships for causality analysis
        - Agent grouping for role analysis
        - Tool call/result pairings
        - Temporal order
        """
        lines = ["=== MAS EXECUTION TRACE ===", ""]
        
        # Basic trace view (sequential)
        lines.append("## Sequential Execution Log")
        for i, span in enumerate(spans):
            parent_info = ""
            if span.parent_span_id:
                # Find parent span's step number
                for j, s in enumerate(spans):
                    if s.span_id == span.parent_span_id:
                        parent_info = f" (follows Step {j+1})"
                        break
            
            lines.append(f"[Step {i+1}]{parent_info} Agent: {span.agent_name}")
            lines.append(f"  Type: {span.step_type.value}")
            content = span.content[:400] if span.content else 'N/A'
            # Indicate if truncated
            if span.content and len(span.content) > 400:
                content += "..."
            lines.append(f"  Content: {content}")
            
            # Show classification confidence if available
            conf = span.attributes.get('classification_confidence')
            if conf is not None:
                lines.append(f"  Classification Confidence: {conf:.2f}")
            lines.append("")
        
        # Agent-grouped view for role analysis
        lines.append("\n## Agent Role Summary")
        agent_spans = {}
        for span in spans:
            agent_spans.setdefault(span.agent_name, []).append(span)
        
        for agent, agent_span_list in agent_spans.items():
            lines.append(f"\n### Agent: {agent}")
            thought_count = sum(1 for s in agent_span_list if s.step_type.value == 'thought')
            action_count = sum(1 for s in agent_span_list if s.step_type.value in ['action', 'tool_call'])
            output_count = sum(1 for s in agent_span_list if s.step_type.value == 'output')
            error_count = sum(1 for s in agent_span_list if s.step_type.value == 'error')
            
            lines.append(f"  Activities: {len(agent_span_list)} total ({thought_count} thoughts, {action_count} actions, {output_count} outputs, {error_count} errors)")
            
            # Show key outputs from this agent
            outputs = [s for s in agent_span_list if s.step_type.value == 'output']
            if outputs:
                lines.append("  Key outputs:")
                for out in outputs[:2]:  # Max 2 outputs
                    lines.append(f"    - {out.content[:200]}...")
        
        # Tool call pairings
        lines.append("\n## Tool Usage Pattern")
        tool_calls = [s for s in spans if s.step_type.value == 'tool_call']
        
        if tool_calls:
            lines.append(f"  Total tool calls: {len(tool_calls)}")
            for tc in tool_calls[:5]:  # Max 5 tool calls
                func_name = tc.attributes.get('function_name', 'unknown')
                lines.append(f"  - {tc.agent_name} called {func_name}")
        else:
            lines.append("  No tool calls in trace")
        
        # Error summary if any
        errors = [s for s in spans if s.step_type.value == 'error']
        if errors:
            lines.append("\n## Errors Detected")
            for err in errors:
                lines.append(f"  - {err.agent_name}: {err.content[:200]}")
        
        return "\n".join(lines)
    
    def _build_prompt(self, trace_text: str) -> str:
        """Build the classification prompt."""
        taxonomy_context = self.taxonomy.to_prompt_context()
        
        if self.mode == ClassifierMode.FEW_SHOT_ICL:
            return self._few_shot_icl_prompt(trace_text, taxonomy_context)
        elif self.mode == ClassifierMode.FINE_TUNED:
            return self._fine_tuned_prompt(trace_text)
        elif self.mode == ClassifierMode.FEW_SHOT:
            return self._few_shot_prompt(trace_text, taxonomy_context)
        else:
            return self._zero_shot_prompt(trace_text, taxonomy_context)
    
    def _few_shot_icl_prompt(self, trace_text: str, taxonomy: str) -> str:
        """
        Build in-context learning prompt with real dataset examples.
        
        Uses the MAD (Multi-Agent Traces) dataset for few-shot learning.
        This provides real-world examples of failure modes for better classification.
        """
        detailed_taxonomy = self._get_detailed_taxonomy()
        
        # Get real examples from MAD dataset for in-context learning
        real_examples = ""
        if self._fewshot_loader:
            real_examples = self._fewshot_loader.get_few_shot_prompt_section(max_examples=5)
        
        if real_examples:
            examples_section = f"""
## REAL EXAMPLES FROM MAD DATASET
The following examples are from the Multi-Agent Systems Traces Dataset (MAD),
showing real failure annotations:

{real_examples}
"""
        else:
            examples_section = """
## EXAMPLE ANALYSIS
Example 1: Agent A ignores results from Agent B -> INTER-5 (Output Ignored)
Example 2: Same step repeated 3+ times -> SPEC-3 (Step Repetition)
Example 3: Agent changes topic without justification -> INTER-3 (Task Derailment)
"""
        
        return f"""You are an expert Multi-Agent System (MAS) failure analyst with deep expertise in the MAST taxonomy.
Your task is to carefully analyze agent execution traces and identify failure modes with high precision.

## MAST TAXONOMY REFERENCE
{detailed_taxonomy}

## ANALYSIS METHODOLOGY
Follow this chain-of-thought analysis process:

### Step 1: Trace Overview
- Identify all agents involved and their roles
- Understand the overall task objective
- Map the flow of information between agents

### Step 2: Per-Agent Analysis
For each agent, examine:
- What was the agent's intended role/responsibility?
- What did the agent actually do?
- Did the agent build upon previous agents' work?
- Were there any deviations from expected behavior?

### Step 3: Inter-Agent Communication Analysis
Examine interactions between agents:
- Was information properly passed between agents?
- Did agents acknowledge and use each other's outputs?
- Were there any communication breakdowns?
- Did any agent deviate from the shared task?

### Step 4: Task Completion Analysis
- Was the original task fully addressed?
- Were there premature terminations?
- Were completion conditions properly recognized?

### Step 5: Pattern Matching to Failure Modes
For each potential issue found, match to specific MAST failure codes:
- Cite specific evidence (quote relevant text)
- Assign confidence based on evidence strength:
  - 0.9-1.0: Clear, unambiguous evidence
  - 0.7-0.9: Strong evidence with minor ambiguity
  - 0.5-0.7: Moderate evidence, some inference required
  - Below 0.5: Do not report

{examples_section}

## TRACE TO ANALYZE
{trace_text}

## OUTPUT FORMAT
Respond with ONLY valid JSON in this exact format:
{{
    "analysis": {{
        "agents_identified": ["list of agent names"],
        "task_objective": "inferred main objective",
        "key_observations": ["observation 1", "observation 2"]
    }},
    "failures": [
        {{
            "code": "FAILURE-CODE",
            "confidence_breakdown": {{
                "evidence_strength": 0.9,
                "pattern_match_score": 0.85,
                "cross_reference_score": 0.7,
                "model_confidence": 0.88
            }},
            "proof_quotes": [
                "Exact quoted text from trace that proves this failure"
            ],
            "affected_steps": ["Step 1", "Step 3"],
            "reasoning_chain": "Step-by-step explanation: 1) First observation, 2) Second observation, 3) Conclusion",
            "evidence": "Summary of evidence",
            "affected_agents": ["agent names involved"]
        }}
    ],
    "summary": "Overall assessment of MAS execution quality"
}}

IMPORTANT:
- Only report failures with final probability >= {self.confidence_threshold}
- Provide EXACT QUOTED TEXT from the trace in proof_quotes
- Reference specific step numbers in affected_steps
- Be thorough but avoid false positives"""
    
    def _fine_tuned_prompt(self, trace_text: str) -> str:
        """
        Build prompt for a real fine-tuned model.
        
        Fine-tuned models already know the taxonomy and task, so only
        minimal instruction is needed to save tokens.
        """
        return f"""Analyze the MAS trace for MAST failure modes.

Trace:
{trace_text}

Output JSON:"""
    
    def _get_detailed_taxonomy(self) -> str:
        """Get detailed taxonomy context for fine-tuned classification."""
        lines = ["### SPECIFICATION ISSUES"]
        
        spec_modes = self.taxonomy.get_specification_modes()
        for mode in spec_modes:
            lines.append(f"\n**[{mode.code}] {mode.name}**")
            lines.append(f"Description: {mode.description}")
            lines.append(f"Severity: {mode.severity.upper()} | Frequency: {mode.frequency}%")
            lines.append("Key Indicators:")
            for indicator in mode.indicators:
                lines.append(f"  • {indicator}")
        
        lines.append("\n### INTER-AGENT MISALIGNMENT")
        inter_modes = self.taxonomy.get_inter_agent_modes()
        for mode in inter_modes:
            lines.append(f"\n**[{mode.code}] {mode.name}**")
            lines.append(f"Description: {mode.description}")
            lines.append(f"Severity: {mode.severity.upper()} | Frequency: {mode.frequency}%")
            lines.append("Key Indicators:")
            for indicator in mode.indicators:
                lines.append(f"  • {indicator}")
        
        lines.append("\n### TASK VERIFICATION")
        task_modes = self.taxonomy.get_task_verification_modes()
        for mode in task_modes:
            lines.append(f"\n**[{mode.code}] {mode.name}**")
            lines.append(f"Description: {mode.description}")
            lines.append(f"Severity: {mode.severity.upper()} | Frequency: {mode.frequency}%")
            lines.append("Key Indicators:")
            for indicator in mode.indicators:
                lines.append(f"  • {indicator}")
        
        return "\n".join(lines)
    
    def _zero_shot_prompt(self, trace_text: str, taxonomy: str) -> str:
        """Build zero-shot classification prompt."""
        return f"""You are an expert Multi-Agent System analyzer trained on the MAST (Multi-Agent System Failure Taxonomy) framework.

{taxonomy}

TASK:
Analyze the following MAS execution trace and identify any failure modes present.

{trace_text}

INSTRUCTIONS:
1. Carefully read the entire trace
2. For each failure mode detected, provide:
   - The failure code (e.g., INTER-3)
   - Confidence score (0.0 to 1.0)
   - Specific evidence from the trace

OUTPUT FORMAT (JSON):
{{
    "failures": [
        {{
            "code": "INTER-3",
            "confidence": 0.85,
            "evidence": "Agent B ignored the research topic specified by Agent A..."
        }}
    ],
    "summary": "Brief summary of trace quality and detected issues"
}}

Only report failures with confidence >= {self.confidence_threshold}.
Respond with valid JSON only."""
    
    def _few_shot_prompt(self, trace_text: str, taxonomy: str) -> str:
        """Build few-shot classification prompt with examples."""
        examples = """
EXAMPLE 1:
Trace: Agent A says "Let's research AI". Agent B responds "I'll write about climate change."
Output: {{"failures": [{{"code": "INTER-3", "confidence": 0.9, "evidence": "Agent B derailed from AI topic to climate change"}}], "summary": "Task derailment detected"}}

EXAMPLE 2:
Trace: Agent A: "x=5". Agent B: "Calculating...". Agent A: "x=5". Agent B: "Still calculating..." Agent A: "x=5"
Output: {{"failures": [{{"code": "SPEC-3", "confidence": 0.95, "evidence": "Agent A repeated 'x=5' three times"}}], "summary": "Step repetition detected"}}

EXAMPLE 3:
Trace: Researcher: "Found 3 key papers on topic". Writer: "I'll write about something else entirely."
Output: {{"failures": [{{"code": "INTER-5", "confidence": 0.88, "evidence": "Writer ignored Researcher's findings"}}], "summary": "Agent output ignored"}}
"""
        
        return f"""You are an expert Multi-Agent System analyzer trained on the MAST framework.

{taxonomy}

{examples}

NOW ANALYZE THIS TRACE:
{trace_text}

OUTPUT FORMAT (JSON):
{{
    "failures": [
        {{
            "code": "CODE",
            "confidence": 0.0-1.0,
            "evidence": "specific evidence"
        }}
    ],
    "summary": "brief summary"
}}

Only report failures with confidence >= {self.confidence_threshold}.
Respond with valid JSON only."""
    
    def _parse_response(self, response_text: str, spans: List[Span]) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        import json
        import re
        
        # Extract JSON from response
        try:
            # Try to find JSON in response - handle markdown code blocks
            response_clean = response_text.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"failures": [], "summary": response_text}
        except json.JSONDecodeError:
            data = {"failures": [], "summary": response_text}
        
        # Convert to FailureMode objects
        failure_modes = []
        for f in data.get("failures", []):
            mode_def = self.taxonomy.get_mode(f.get("code", ""))
            if not mode_def:
                continue
            
            # Parse confidence breakdown if available
            conf_breakdown_data = f.get("confidence_breakdown", {})
            if conf_breakdown_data:
                conf_breakdown = ConfidenceBreakdown(
                    evidence_strength=conf_breakdown_data.get("evidence_strength", 0.5),
                    pattern_match_score=conf_breakdown_data.get("pattern_match_score", 0.5),
                    cross_reference_score=conf_breakdown_data.get("cross_reference_score", 0.5),
                    model_confidence=conf_breakdown_data.get("model_confidence", 0.5)
                )
                final_prob = conf_breakdown.final_probability
            else:
                # Fallback: use raw confidence and create breakdown from calibration
                raw_conf = f.get("confidence", 0.5)
                evidence = f.get("evidence", "")
                calibrated = self._calibrate_confidence(raw_conf, evidence)
                conf_breakdown = ConfidenceBreakdown(
                    evidence_strength=calibrated,
                    pattern_match_score=raw_conf,
                    cross_reference_score=raw_conf * 0.8,
                    model_confidence=raw_conf
                )
                final_prob = conf_breakdown.final_probability
            
            # Filter by threshold
            if final_prob < self.confidence_threshold:
                continue
            
            # Extract step references
            affected_steps = f.get("affected_steps", []) or f.get("span_references", [])
            affected_span_ids = []
            
            for ref in affected_steps:
                step_match = re.search(r'Step\s+(\d+)', str(ref), re.IGNORECASE)
                if step_match:
                    idx = int(step_match.group(1)) - 1
                    if 0 <= idx < len(spans):
                        affected_span_ids.append(spans[idx].span_id)
            
            # Fall back to first few spans if no matches
            if not affected_span_ids:
                affected_span_ids = [s.span_id for s in spans[:3]]
            
            # Extract proof quotes
            proof_quotes = f.get("proof_quotes", [])
            if not proof_quotes and f.get("evidence"):
                # Create proof quote from evidence
                proof_quotes = [f.get("evidence", "")[:200]]
            
            # Build evidence summary
            evidence = f.get("evidence", "")
            reasoning = f.get("reasoning_chain", "") or f.get("reasoning", "")
            if reasoning and reasoning not in evidence:
                evidence = f"{evidence}\nReasoning: {reasoning}"
            
            failure_modes.append(FailureMode(
                code=f["code"],
                name=mode_def.name,
                category=mode_def.category.value,
                confidence=final_prob,  # Use computed probability
                evidence=evidence,
                span_ids=affected_span_ids,
                # New enhanced fields
                confidence_breakdown=conf_breakdown,
                proof_quotes=proof_quotes,
                affected_steps=affected_steps,
                reasoning_chain=reasoning
            ))
        
        # Extract analysis if available
        analysis = data.get("analysis", {})
        summary = data.get("summary", "")
        if analysis and not summary:
            summary = f"Analyzed {len(analysis.get('agents_identified', []))} agents. "
            if analysis.get("key_observations"):
                summary += "Key: " + "; ".join(analysis["key_observations"][:2])
        
        return ClassificationResult(
            failure_modes=failure_modes,
            trace_summary=summary,
            confidence_threshold=self.confidence_threshold,
            model_used=self.model_name
        )
    
    def _calibrate_confidence(self, raw_confidence: float, evidence: str) -> float:
        """
        Calibrate confidence score based on evidence quality.
        
        Args:
            raw_confidence: The model's reported confidence
            evidence: The evidence string provided
            
        Returns:
            Calibrated confidence score
        """
        # Start with raw confidence
        calibrated = raw_confidence
        
        # Boost confidence if evidence is detailed (>100 chars)
        if len(evidence) > 100:
            calibrated = min(1.0, calibrated + 0.05)
        
        # Reduce confidence if evidence is sparse
        if len(evidence) < 30:
            calibrated = max(0.0, calibrated - 0.1)
        
        # Reduce if evidence seems generic
        generic_phrases = ["may have", "possibly", "could be", "might"]
        if any(phrase in evidence.lower() for phrase in generic_phrases):
            calibrated = max(0.0, calibrated - 0.15)
        
        return round(calibrated, 2)
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported failure mode codes."""
        return self.taxonomy.get_codes()
    
    def classify_batch(self, traces: List[List[Span]]) -> List[ClassificationResult]:
        """
        Classify multiple traces.
        
        Args:
            traces: List of span lists
        
        Returns:
            List of ClassificationResults
        """
        return [self.classify(spans) for spans in traces]
    
    def get_taxonomy(self) -> MASTTaxonomy:
        """Get the taxonomy used by this classifier."""
        return self.taxonomy
    
    def summary(self, result: ClassificationResult) -> str:
        """Generate human-readable summary of classification."""
        lines = [
            "=" * 50,
            "MAST CLASSIFICATION RESULTS",
            "=" * 50,
            "",
            f"Model: {result.model_used}",
            f"Failures Detected: {len(result.failure_modes)}",
            ""
        ]
        
        if result.failure_modes:
            lines.append("Detected Failures:")
            for f in result.failure_modes:
                lines.append(f"  [{f.code}] {f.name}")
                
                # Show probability with breakdown if available
                if f.confidence_breakdown:
                    prob = f.confidence_breakdown.final_probability
                    lines.append(f"    🎯 Probability: {prob * 100:.1f}%")
                    lines.append(f"       └─ Evidence Strength:    {f.confidence_breakdown.evidence_strength:.0%}")
                    lines.append(f"       └─ Pattern Match:        {f.confidence_breakdown.pattern_match_score:.0%}")
                    lines.append(f"       └─ Cross-Reference:      {f.confidence_breakdown.cross_reference_score:.0%}")
                    lines.append(f"       └─ Model Confidence:     {f.confidence_breakdown.model_confidence:.0%}")
                else:
                    lines.append(f"    Confidence: {f.confidence:.0%}")
                
                lines.append(f"    Category: {f.category}")
                
                # Show proof quotes if available
                if f.proof_quotes:
                    lines.append("    📝 Proof Quotes:")
                    for quote in f.proof_quotes[:2]:
                        lines.append(f"       \"{quote[:100]}...\"")
                
                # Show affected steps if available
                if f.affected_steps:
                    lines.append(f"    📍 Affected Steps: {', '.join(f.affected_steps[:5])}")
                
                # Show reasoning chain if available
                if f.reasoning_chain:
                    lines.append(f"    💭 Reasoning: {f.reasoning_chain[:150]}...")
                
                lines.append("")
        else:
            lines.append("✅ No failure modes detected above threshold")
        
        lines.append("")
        lines.append(f"Summary: {result.trace_summary}")
        
        return "\n".join(lines)

