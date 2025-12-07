"""
Thought Relevance Score (TRS) Metric.

Measures how relevant each agent's thoughts are to the task objective
and final outcome. Helps identify agents with useful vs. irrelevant reasoning.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import networkx as nx

from mas_eval.core.base import BaseMetric
from mas_eval.core.types import Span, StepType


class ThoughtRelevanceMetric(BaseMetric):
    """
    Thought Relevance Score (TRS) Metric.
    
    Measures the semantic relevance of agent thoughts to:
    1. The original task/goal
    2. The final output/result
    3. The flow of conversation
    
    High TRS = Thoughts are focused and relevant (Good)
    Low TRS = Thoughts are tangential or off-topic (Bad)
    
    Usage:
        metric = ThoughtRelevanceMetric()
        result = metric.calculate(graph, spans, task_description="Research AI")
        print(f"TRS: {result['overall_score']}")
        print(f"Per-agent: {result['agent_scores']}")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the TRS metric.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        super().__init__(
            name="TRS",
            description="Thought Relevance Score - measures usefulness of agent reasoning"
        )
        self.model_name = model_name
        self._encoder = None
    
    def _load_encoder(self):
        """Lazy load the sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for TRS. "
                    "Install with: pip install sentence-transformers"
                )
    
    def calculate(
        self,
        graph: nx.DiGraph = None,
        spans: List[Span] = None,
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate Thought Relevance Score.
        
        Args:
            graph: Causal Reasoning Graph (optional, not used directly)
            spans: List of spans to analyze
            task_description: Optional description of the original task
            
        Returns:
            Dictionary with overall score and per-agent breakdown
        """
        if not spans:
            return {
                "overall_score": 1.0,
                "agent_scores": {},
                "thought_count": 0,
                "interpretation": "No spans to analyze"
            }
        
        # Extract thought spans
        thought_spans = [s for s in spans if s.step_type == StepType.THOUGHT]
        
        if not thought_spans:
            # Check if thoughts might be labeled differently
            thought_spans = [s for s in spans if "thought" in s.name.lower()]
            
        if not thought_spans:
            return {
                "overall_score": 1.0,
                "agent_scores": {},
                "thought_count": 0,
                "interpretation": "No thought spans found in trace"
            }
        
        # Extract output spans for reference
        output_spans = [
            s for s in spans 
            if s.step_type == StepType.OUTPUT or "output" in s.name.lower()
        ]
        
        self._load_encoder()
        
        # Build reference content for comparison
        reference_texts = self._build_reference_texts(
            spans, task_description, output_spans
        )
        
        # Calculate relevance for each thought
        agent_scores = {}
        thought_scores = []
        
        for thought_span in thought_spans:
            if not thought_span.content or len(thought_span.content.strip()) < 5:
                continue
                
            score = self._calculate_thought_relevance(
                thought_span.content,
                reference_texts
            )
            thought_scores.append(score)
            
            # Aggregate by agent
            agent = thought_span.agent_name
            if agent not in agent_scores:
                agent_scores[agent] = {"scores": [], "count": 0}
            agent_scores[agent]["scores"].append(score)
            agent_scores[agent]["count"] += 1
        
        # Calculate agent averages
        agent_averages = {}
        for agent, data in agent_scores.items():
            if data["scores"]:
                avg = np.mean(data["scores"])
                agent_averages[agent] = {
                    "score": round(avg, 4),
                    "thought_count": data["count"],
                    "interpretation": self._interpret_score(avg)
                }
        
        # Calculate overall score
        if thought_scores:
            overall_score = round(np.mean(thought_scores), 4)
        else:
            overall_score = 1.0
        
        return {
            "overall_score": overall_score,
            "agent_scores": agent_averages,
            "thought_count": len(thought_scores),
            "interpretation": self.get_interpretation(overall_score)
        }
    
    def _build_reference_texts(
        self,
        spans: List[Span],
        task_description: Optional[str],
        output_spans: List[Span]
    ) -> List[str]:
        """Build reference texts for relevance comparison."""
        references = []
        
        # Add task description if provided
        if task_description:
            references.append(task_description)
        else:
            # Try to extract task from first span
            first_content = spans[0].content if spans else ""
            if first_content:
                references.append(first_content[:500])
        
        # Add output content as references
        for output in output_spans:
            if output.content:
                references.append(output.content[:500])
        
        # Add action spans as context
        action_spans = [
            s for s in spans 
            if s.step_type == StepType.ACTION or "action" in s.name.lower()
        ]
        for action in action_spans[:3]:  # Limit to first 3 actions
            if action.content:
                references.append(action.content[:300])
        
        return references
    
    def _calculate_thought_relevance(
        self,
        thought_content: str,
        reference_texts: List[str]
    ) -> float:
        """
        Calculate relevance of a single thought to reference texts.
        
        Returns similarity score between 0 and 1.
        """
        if not reference_texts:
            return 0.5  # Neutral if no references
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get embeddings
            thought_embedding = self._encoder.encode([thought_content])
            reference_embeddings = self._encoder.encode(reference_texts)
            
            # Calculate similarity to each reference
            similarities = cosine_similarity(thought_embedding, reference_embeddings)[0]
            
            # Return max similarity (most relevant reference)
            # Using max because a thought only needs to be relevant to one aspect
            max_sim = np.max(similarities)
            
            # Also consider average for overall coherence
            avg_sim = np.mean(similarities)
            
            # Weighted combination (80% max, 20% avg)
            # This rewards thoughts that are highly relevant to at least one reference
            # while still considering overall coherence
            combined = 0.8 * max_sim + 0.2 * avg_sim
            
            return float(combined)
            
        except Exception as e:
            print(f"Warning: Error calculating thought relevance: {e}")
            return 0.5
    
    def _interpret_score(self, score: float) -> str:
        """Interpret a single score."""
        if score >= 0.7:
            return "Highly relevant"
        elif score >= 0.5:
            return "Moderately relevant"
        elif score >= 0.3:
            return "Somewhat relevant"
        else:
            return "Low relevance"
    
    def get_interpretation(self, value: float) -> str:
        """Interpret overall TRS value."""
        if value >= 0.7:
            return "Excellent - Agent thoughts are highly focused and relevant to the task"
        elif value >= 0.5:
            return "Good - Agent thoughts are mostly on-topic with some tangents"
        elif value >= 0.3:
            return "Fair - Agent thoughts show moderate relevance, some off-topic reasoning"
        else:
            return "Poor - Agent thoughts appear largely irrelevant to the task"
    
    def get_improvement_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """
        Generate suggestions for improving thought relevance.
        
        Args:
            result: Result from calculate()
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        overall = result.get("overall_score", 1.0)
        agent_scores = result.get("agent_scores", {})
        
        if overall < 0.5:
            suggestions.append(
                "Consider adding explicit task anchoring in agent system prompts "
                "to keep thoughts focused on the main objective"
            )
        
        if overall < 0.3:
            suggestions.append(
                "Agent thoughts appear largely off-topic. Review agent instructions "
                "and ensure they understand their specific role in the MAS"
            )
        
        # Find underperforming agents
        for agent, data in agent_scores.items():
            if data.get("score", 1.0) < 0.4:
                suggestions.append(
                    f"Agent '{agent}' shows low thought relevance ({data['score']:.2f}). "
                    f"Consider refining its role description and constraints"
                )
        
        # Check for imbalanced agents
        if agent_scores:
            scores = [d.get("score", 0) for d in agent_scores.values()]
            if max(scores) - min(scores) > 0.4:
                suggestions.append(
                    "Significant variation in thought relevance across agents. "
                    "Consider standardizing prompt structures for consistency"
                )
        
        return suggestions


class ThoughtQualityMetric(BaseMetric):
    """
    Extended thought quality analysis beyond just relevance.
    
    Measures:
    - Relevance (TRS)
    - Coherence (internal consistency)
    - Depth (level of analysis)
    - Actionability (leads to concrete outputs)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(
            name="ThoughtQuality",
            description="Comprehensive thought quality assessment"
        )
        self.trs = ThoughtRelevanceMetric(model_name)
    
    def calculate(
        self,
        graph: nx.DiGraph = None,
        spans: List[Span] = None,
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive thought quality metrics."""
        # Get base TRS
        trs_result = self.trs.calculate(graph, spans, task_description)
        
        # Calculate additional metrics
        coherence = self._calculate_coherence(spans)
        depth = self._calculate_depth(spans)
        actionability = self._calculate_actionability(spans)
        
        # Weighted overall score
        overall = (
            0.4 * trs_result["overall_score"] +
            0.2 * coherence +
            0.2 * depth +
            0.2 * actionability
        )
        
        return {
            "overall_score": round(overall, 4),
            "relevance": trs_result["overall_score"],
            "coherence": round(coherence, 4),
            "depth": round(depth, 4),
            "actionability": round(actionability, 4),
            "agent_details": trs_result["agent_scores"],
            "interpretation": self.get_interpretation(overall)
        }
    
    def _calculate_coherence(self, spans: List[Span]) -> float:
        """Calculate coherence of thought progression."""
        thought_spans = [
            s for s in spans 
            if s.step_type == StepType.THOUGHT or "thought" in s.name.lower()
        ]
        
        if len(thought_spans) < 2:
            return 1.0  # Single thought is coherent by default
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.trs._load_encoder()
            contents = [s.content for s in thought_spans if s.content]
            
            if len(contents) < 2:
                return 1.0
            
            embeddings = self.trs._encoder.encode(contents)
            
            # Calculate sequential coherence (consecutive thoughts should relate)
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append(sim)
            
            return float(np.mean(coherence_scores))
            
        except Exception:
            return 0.5
    
    def _calculate_depth(self, spans: List[Span]) -> float:
        """Estimate depth of analysis based on thought complexity."""
        thought_spans = [
            s for s in spans 
            if s.step_type == StepType.THOUGHT or "thought" in s.name.lower()
        ]
        
        if not thought_spans:
            return 0.5
        
        # Heuristics for depth:
        # - Longer thoughts (up to a point)
        # - Use of analytical language
        # - Structured reasoning indicators
        
        depth_indicators = [
            "because", "therefore", "however", "considering",
            "analysis", "compare", "evaluate", "conclude",
            "first", "second", "finally", "step",
            "pros", "cons", "tradeoff", "implication"
        ]
        
        scores = []
        for span in thought_spans:
            content = span.content.lower() if span.content else ""
            
            # Length score (0-0.5)
            length_score = min(1.0, len(content) / 500) * 0.5
            
            # Indicator score (0-0.5)
            indicator_count = sum(1 for ind in depth_indicators if ind in content)
            indicator_score = min(1.0, indicator_count / 4) * 0.5
            
            scores.append(length_score + indicator_score)
        
        return float(np.mean(scores))
    
    def _calculate_actionability(self, spans: List[Span]) -> float:
        """Measure how well thoughts lead to actions/outputs."""
        thought_spans = [
            (i, s) for i, s in enumerate(spans)
            if s.step_type == StepType.THOUGHT or "thought" in s.name.lower()
        ]
        
        if not thought_spans:
            return 0.5
        
        # Check if thoughts are followed by actions/outputs
        actionable_count = 0
        for idx, thought in thought_spans:
            # Look at next few spans
            for j in range(idx + 1, min(idx + 4, len(spans))):
                next_span = spans[j]
                if next_span.step_type in [StepType.ACTION, StepType.OUTPUT, StepType.TOOL_CALL]:
                    actionable_count += 1
                    break
        
        return actionable_count / len(thought_spans) if thought_spans else 0.5
    
    def get_interpretation(self, value: float) -> str:
        if value >= 0.7:
            return "Excellent thought quality across all dimensions"
        elif value >= 0.5:
            return "Good thought quality with room for improvement"
        elif value >= 0.3:
            return "Fair thought quality - consider improving agent reasoning"
        else:
            return "Poor thought quality - significant improvements needed"
