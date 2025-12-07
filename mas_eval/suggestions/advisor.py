"""
MAS Advisor - Generates improvement suggestions for Multi-Agent Systems.

Analyzes evaluation results and provides actionable recommendations to
improve MAS performance based on detected failures and metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from mas_eval.core.types import EvaluationResult, FailureMode


class SuggestionCategory(Enum):
    """Categories of improvement suggestions."""
    PROMPT_ENGINEERING = "prompt_engineering"
    ARCHITECTURE = "architecture"
    COMMUNICATION = "communication"
    TASK_DESIGN = "task_design"
    MONITORING = "monitoring"
    GUARDRAILS = "guardrails"


class Priority(Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    """
    A single improvement suggestion.
    
    Attributes:
        title: Short title for the suggestion
        description: Detailed description of what to do
        category: Category of the suggestion
        priority: Priority level
        related_failures: Failure codes this addresses
        implementation_hints: Specific implementation guidance
        expected_impact: Expected improvement from implementing this
    """
    title: str
    description: str
    category: SuggestionCategory
    priority: Priority
    related_failures: List[str] = field(default_factory=list)
    implementation_hints: List[str] = field(default_factory=list)
    expected_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "related_failures": self.related_failures,
            "implementation_hints": self.implementation_hints,
            "expected_impact": self.expected_impact
        }


class MASAdvisor:
    """
    Generates improvement suggestions for Multi-Agent Systems.
    
    Analyzes evaluation results including:
    - MAST failure modes
    - GEMMAS metrics (IDS, UPR)
    - Thought Relevance Score (TRS)
    - Graph statistics
    
    And produces actionable recommendations.
    
    Usage:
        advisor = MASAdvisor()
        suggestions = advisor.generate_suggestions(evaluation_result)
        
        for suggestion in suggestions:
            print(f"[{suggestion.priority.value}] {suggestion.title}")
            print(f"  {suggestion.description}")
    """
    
    # Failure-specific suggestions database
    FAILURE_SUGGESTIONS = {
        "SPEC-1": {
            "title": "Improve Task Specification Compliance",
            "description": "Agents are not following task specifications. Strengthen explicit constraints in prompts.",
            "category": SuggestionCategory.PROMPT_ENGINEERING,
            "hints": [
                "Add explicit 'REQUIREMENTS' section in agent prompts",
                "Use structured output formats with validation",
                "Add examples of correct vs incorrect outputs",
                "Consider adding a validator agent to check compliance"
            ],
            "impact": "Reduces task specification violations by enforcing clearer requirements"
        },
        "SPEC-2": {
            "title": "Clarify Agent Role Boundaries",
            "description": "Agents are stepping outside their defined roles. Need stronger role definitions.",
            "category": SuggestionCategory.ARCHITECTURE,
            "hints": [
                "Define explicit 'DO' and 'DON'T' sections in role descriptions",
                "Add examples of role-appropriate vs inappropriate actions",
                "Implement role-based access control for tools",
                "Consider adding a supervisor agent to enforce boundaries"
            ],
            "impact": "Prevents role confusion and improves agent specialization"
        },
        "SPEC-3": {
            "title": "Eliminate Step Repetition",
            "description": "Agents are repeating steps unnecessarily. Add deduplication logic.",
            "category": SuggestionCategory.GUARDRAILS,
            "hints": [
                "Track completed steps in agent state",
                "Add 'already done' checks before actions",
                "Implement step caching mechanism",
                "Set maximum iteration limits"
            ],
            "impact": "Reduces redundant operations and improves efficiency"
        },
        "SPEC-4": {
            "title": "Preserve Conversation Context",
            "description": "Context is being lost during execution. Improve context management.",
            "category": SuggestionCategory.ARCHITECTURE,
            "hints": [
                "Implement explicit context summarization between steps",
                "Use structured memory/state objects",
                "Add context passing in agent calls",
                "Consider using a shared memory store"
            ],
            "impact": "Maintains coherent multi-turn conversations and decisions"
        },
        "SPEC-5": {
            "title": "Define Clear Termination Conditions",
            "description": "Agents don't know when to stop. Add explicit completion criteria.",
            "category": SuggestionCategory.TASK_DESIGN,
            "hints": [
                "Define explicit success criteria in task description",
                "Add 'is_complete' check in agent logic",
                "Set maximum step limits as fallback",
                "Train agents to recognize completion signals"
            ],
            "impact": "Prevents infinite loops and ensures clean task completion"
        },
        "INTER-1": {
            "title": "Prevent Conversation Resets",
            "description": "Agents are resetting context unexpectedly. Add continuity safeguards.",
            "category": SuggestionCategory.COMMUNICATION,
            "hints": [
                "Implement conversation state persistence",
                "Add context summaries at conversation start",
                "Use explicit handoff protocols between agents",
                "Log and monitor context transitions"
            ],
            "impact": "Maintains conversation continuity and prevents lost progress"
        },
        "INTER-2": {
            "title": "Encourage Clarification Requests",
            "description": "Agents proceed with ambiguous instructions. Train them to ask questions.",
            "category": SuggestionCategory.PROMPT_ENGINEERING,
            "hints": [
                "Add 'ask for clarification when uncertain' to instructions",
                "Provide examples of when to clarify",
                "Implement confidence thresholds for actions",
                "Create a clarification request protocol"
            ],
            "impact": "Reduces errors from misunderstood requirements"
        },
        "INTER-3": {
            "title": "Add Task Anchoring Mechanisms",
            "description": "Agents are derailing from the main task. Add focus mechanisms.",
            "category": SuggestionCategory.GUARDRAILS,
            "hints": [
                "Include task summary in every agent prompt",
                "Add periodic task-relevance checks",
                "Implement topic drift detection",
                "Use explicit goal reminders between steps"
            ],
            "impact": "Keeps agents focused on the original objective"
        },
        "INTER-4": {
            "title": "Improve Information Sharing",
            "description": "Agents are withholding relevant information. Enhance handoff protocols.",
            "category": SuggestionCategory.COMMUNICATION,
            "hints": [
                "Define explicit output schemas for each agent",
                "Require structured handoff objects",
                "Add 'summary' field in agent outputs",
                "Implement information completeness checks"
            ],
            "impact": "Ensures all relevant information flows between agents"
        },
        "INTER-5": {
            "title": "Enforce Output Acknowledgment",
            "description": "Agents are ignoring other agents' outputs. Add acknowledgment requirements.",
            "category": SuggestionCategory.COMMUNICATION,
            "hints": [
                "Require agents to reference previous outputs explicitly",
                "Add 'based on X's findings' in prompts",
                "Implement output injection in downstream prompts",
                "Add verification that outputs were considered"
            ],
            "impact": "Ensures collaborative use of all agent contributions"
        },
        "INTER-6": {
            "title": "Align Reasoning with Actions",
            "description": "Agent reasoning doesn't match their actions. Add consistency checks.",
            "category": SuggestionCategory.MONITORING,
            "hints": [
                "Require explicit reasoning before actions",
                "Add action-reasoning validation step",
                "Log and compare stated intentions vs actual actions",
                "Implement chain-of-thought consistency checks"
            ],
            "impact": "Reduces unpredictable behavior and improves transparency"
        },
        "TASK-1": {
            "title": "Prevent Premature Termination",
            "description": "Tasks are ending before completion. Add completion verification.",
            "category": SuggestionCategory.TASK_DESIGN,
            "hints": [
                "Define explicit completion checklist",
                "Add final verification step before output",
                "Implement 'completeness score' metric",
                "Require all subtasks to be addressed"
            ],
            "impact": "Ensures tasks are fully completed before termination"
        },
        "TASK-2": {
            "title": "Improve Task Completion Detection",
            "description": "Agents can't recognize when tasks are done. Add clear signals.",
            "category": SuggestionCategory.TASK_DESIGN,
            "hints": [
                "Define measurable success criteria",
                "Add explicit 'TASK_COMPLETE' signal protocol",
                "Implement objective-based completion checks",
                "Use structured output for final results"
            ],
            "impact": "Reduces over-processing and enables clean handoffs"
        },
        "TASK-3": {
            "title": "Enhance Failure Detection",
            "description": "Agents don't recognize failures. Add error handling.",
            "category": SuggestionCategory.MONITORING,
            "hints": [
                "Define explicit failure conditions",
                "Add error detection and reporting",
                "Implement graceful degradation",
                "Use try-catch patterns with meaningful error messages"
            ],
            "impact": "Prevents silent failures and enables recovery"
        }
    }
    
    # Metric-based thresholds
    METRIC_THRESHOLDS = {
        "IDS": {"excellent": 0.7, "good": 0.5, "poor": 0.3},
        "UPR": {"excellent": 0.2, "good": 0.4, "poor": 0.6},
        "TRS": {"excellent": 0.7, "good": 0.5, "poor": 0.3}
    }
    
    def __init__(self, include_low_priority: bool = True, model: str = None):
        """
        Initialize the MAS Advisor.
        
        Args:
            include_low_priority: Whether to include low priority suggestions
            model: Optional model parameter (kept for API compatibility, not used)
        """
        self.include_low_priority = include_low_priority
        # model parameter is accepted for backward compatibility but not used
        # (advisor uses rule-based logic, not LLM)
    
    def generate_suggestions(
        self,
        result: EvaluationResult,
        trs_result: Optional[Dict[str, Any]] = None
    ) -> List[Suggestion]:
        """
        Generate improvement suggestions based on evaluation results.
        
        Args:
            result: The evaluation result from MASEvaluator
            trs_result: Optional TRS metric results
            
        Returns:
            List of prioritized Suggestion objects
        """
        suggestions = []
        
        # Generate failure-based suggestions
        suggestions.extend(self._failure_based_suggestions(result.failures))
        
        # Generate metrics-based suggestions
        suggestions.extend(self._metrics_based_suggestions(result.metrics, trs_result))
        
        # Generate graph-based suggestions
        suggestions.extend(self._graph_based_suggestions(result.graph_stats))
        
        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        suggestions.sort(key=lambda s: priority_order[s.priority])
        
        # Filter low priority if requested
        if not self.include_low_priority:
            suggestions = [s for s in suggestions if s.priority != Priority.LOW]
        
        # Deduplicate similar suggestions
        suggestions = self._deduplicate(suggestions)
        
        return suggestions
    
    def generate_suggestions_from_data(
        self,
        spans: List = None,
        failures: List[FailureMode] = None,
        metrics: Dict[str, float] = None,
        graph_stats: Dict[str, Any] = None,
        task_description: str = ""
    ) -> List[Suggestion]:
        """
        Alternative API accepting raw data instead of EvaluationResult.
        
        This method provides backward compatibility for code that passes
        individual arguments rather than an EvaluationResult object.
        
        Args:
            spans: List of spans (not directly used, kept for API compat)
            failures: List of detected failure modes
            metrics: Dictionary of metric values (IDS, UPR, etc.)
            graph_stats: Graph statistics
            task_description: Description of the task (not used)
            
        Returns:
            List of prioritized Suggestion objects
        """
        # Create a minimal EvaluationResult from components
        result = EvaluationResult(
            trace_id="from_data",
            metrics=metrics or {},
            failures=failures or [],
            graph_stats=graph_stats or {}
        )
        return self.generate_suggestions(result)
    
    def _failure_based_suggestions(
        self,
        failures: List[FailureMode]
    ) -> List[Suggestion]:
        """Generate suggestions based on detected failures."""
        suggestions = []
        
        for failure in failures:
            code = failure.code
            if code in self.FAILURE_SUGGESTIONS:
                data = self.FAILURE_SUGGESTIONS[code]
                
                # Determine priority based on failure confidence
                if failure.confidence >= 0.8:
                    priority = Priority.CRITICAL
                elif failure.confidence >= 0.6:
                    priority = Priority.HIGH
                else:
                    priority = Priority.MEDIUM
                
                suggestion = Suggestion(
                    title=data["title"],
                    description=data["description"],
                    category=data["category"],
                    priority=priority,
                    related_failures=[code],
                    implementation_hints=data["hints"],
                    expected_impact=data["impact"]
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _metrics_based_suggestions(
        self,
        metrics: Dict[str, float],
        trs_result: Optional[Dict[str, Any]] = None
    ) -> List[Suggestion]:
        """Generate suggestions based on metric values."""
        suggestions = []
        
        # IDS (Information Diversity Score) suggestions
        ids = metrics.get("IDS", 0.5)
        if ids < self.METRIC_THRESHOLDS["IDS"]["poor"]:
            suggestions.append(Suggestion(
                title="Increase Information Diversity",
                description="Agents are producing repetitive content. Encourage unique contributions.",
                category=SuggestionCategory.PROMPT_ENGINEERING,
                priority=Priority.HIGH,
                implementation_hints=[
                    "Assign more distinct roles to each agent",
                    "Add 'provide unique perspective' to instructions",
                    "Avoid copying previous agent outputs",
                    "Use different information sources per agent"
                ],
                expected_impact="Improves content diversity and reduces redundancy"
            ))
        elif ids < self.METRIC_THRESHOLDS["IDS"]["good"]:
            suggestions.append(Suggestion(
                title="Improve Content Differentiation",
                description="Some redundancy in agent outputs. Consider more specialized roles.",
                category=SuggestionCategory.ARCHITECTURE,
                priority=Priority.MEDIUM,
                implementation_hints=[
                    "Review agent role overlaps",
                    "Ensure each agent adds unique value"
                ],
                expected_impact="Reduces redundant processing"
            ))
        
        # UPR (Unnecessary Path Ratio) suggestions
        upr = metrics.get("UPR", 0.5)
        if upr > self.METRIC_THRESHOLDS["UPR"]["poor"]:
            suggestions.append(Suggestion(
                title="Streamline Reasoning Paths",
                description="Many unnecessary detours in reasoning. Optimize agent coordination.",
                category=SuggestionCategory.ARCHITECTURE,
                priority=Priority.HIGH,
                implementation_hints=[
                    "Review and simplify agent workflow",
                    "Remove redundant intermediate steps",
                    "Use sequential agents for linear tasks",
                    "Consider parallel execution where possible"
                ],
                expected_impact="Reduces wasted computation and improves speed"
            ))
        elif upr > self.METRIC_THRESHOLDS["UPR"]["good"]:
            suggestions.append(Suggestion(
                title="Optimize Reasoning Efficiency",
                description="Some inefficient paths detected. Consider workflow optimization.",
                category=SuggestionCategory.TASK_DESIGN,
                priority=Priority.MEDIUM,
                implementation_hints=[
                    "Analyze which steps can be combined",
                    "Check for unnecessary back-and-forth"
                ],
                expected_impact="Improves execution efficiency"
            ))
        
        # TRS (Thought Relevance Score) suggestions
        if trs_result:
            trs = trs_result.get("overall_score", 0.5)
            if trs < self.METRIC_THRESHOLDS["TRS"]["poor"]:
                suggestions.append(Suggestion(
                    title="Improve Thought Relevance",
                    description="Agent thoughts are not well-aligned with the task. Add focus mechanisms.",
                    category=SuggestionCategory.PROMPT_ENGINEERING,
                    priority=Priority.HIGH,
                    implementation_hints=[
                        "Add explicit task anchoring in prompts",
                        "Include goal reminders in system prompts",
                        "Require thoughts to reference the task",
                        "Add relevance validation steps"
                    ],
                    expected_impact="Keeps agent reasoning focused and productive"
                ))
            elif trs < self.METRIC_THRESHOLDS["TRS"]["good"]:
                suggestions.append(Suggestion(
                    title="Enhance Thought Focus",
                    description="Agent thoughts show moderate relevance. Consider additional guidance.",
                    category=SuggestionCategory.PROMPT_ENGINEERING,
                    priority=Priority.MEDIUM,
                    implementation_hints=[
                        "Review agent instructions for clarity",
                        "Add more specific task context"
                    ],
                    expected_impact="Improves reasoning relevance"
                ))
            
            # Per-agent suggestions
            agent_scores = trs_result.get("agent_scores", {})
            for agent, data in agent_scores.items():
                if data.get("score", 1.0) < 0.4:
                    suggestions.append(Suggestion(
                        title=f"Refocus Agent: {agent}",
                        description=f"Agent '{agent}' shows low thought relevance ({data['score']:.2f}). Needs targeted improvement.",
                        category=SuggestionCategory.PROMPT_ENGINEERING,
                        priority=Priority.MEDIUM,
                        related_failures=[],
                        implementation_hints=[
                            f"Review and refine {agent}'s role description",
                            "Add more specific constraints and examples",
                            "Consider adjusting agent's scope"
                        ],
                        expected_impact=f"Improves {agent}'s contribution quality"
                    ))
        
        return suggestions
    
    def _graph_based_suggestions(
        self,
        graph_stats: Dict[str, Any]
    ) -> List[Suggestion]:
        """Generate suggestions based on graph structure."""
        suggestions = []
        
        num_nodes = graph_stats.get("num_nodes", 0)
        num_agents = graph_stats.get("num_agents", 0)
        
        # Check for potential issues
        if num_nodes > 0 and num_agents > 0:
            nodes_per_agent = num_nodes / num_agents
            
            # Too many nodes per agent might indicate verbosity
            if nodes_per_agent > 10:
                suggestions.append(Suggestion(
                    title="Reduce Agent Verbosity",
                    description="Agents are generating many steps. Consider consolidating outputs.",
                    category=SuggestionCategory.PROMPT_ENGINEERING,
                    priority=Priority.LOW,
                    implementation_hints=[
                        "Ask agents to be concise",
                        "Combine related outputs",
                        "Use structured responses"
                    ],
                    expected_impact="Reduces trace complexity"
                ))
            
            # Too few nodes might indicate shallow processing
            if nodes_per_agent < 2:
                suggestions.append(Suggestion(
                    title="Encourage Deeper Analysis",
                    description="Agents may not be exploring problems thoroughly.",
                    category=SuggestionCategory.PROMPT_ENGINEERING,
                    priority=Priority.LOW,
                    implementation_hints=[
                        "Ask agents to show their reasoning",
                        "Require multi-step analysis",
                        "Add 'think step by step' instructions"
                    ],
                    expected_impact="Improves reasoning depth"
                ))
        
        return suggestions
    
    def _deduplicate(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Remove duplicate or very similar suggestions."""
        seen_titles = set()
        unique = []
        
        for suggestion in suggestions:
            normalized = suggestion.title.lower().strip()
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique.append(suggestion)
        
        return unique
    
    def format_suggestions(
        self,
        suggestions: List[Suggestion],
        format: str = "text"
    ) -> str:
        """
        Format suggestions for display.
        
        Args:
            suggestions: List of suggestions
            format: Output format ("text" or "html")
            
        Returns:
            Formatted string
        """
        if format == "html":
            return self._format_html(suggestions)
        return self._format_text(suggestions)
    
    def _format_text(self, suggestions: List[Suggestion]) -> str:
        """Format suggestions as text."""
        if not suggestions:
            return "✅ No improvement suggestions - MAS appears to be performing well!"
        
        lines = [
            "=" * 60,
            "MAS IMPROVEMENT SUGGESTIONS",
            "=" * 60,
            ""
        ]
        
        current_priority = None
        for suggestion in suggestions:
            if suggestion.priority != current_priority:
                current_priority = suggestion.priority
                priority_emoji = {
                    Priority.CRITICAL: "🔴",
                    Priority.HIGH: "🟠",
                    Priority.MEDIUM: "🟡",
                    Priority.LOW: "🟢"
                }
                lines.append(f"\n{priority_emoji[current_priority]} {current_priority.value.upper()} PRIORITY")
                lines.append("-" * 40)
            
            lines.append(f"\n📌 {suggestion.title}")
            lines.append(f"   Category: {suggestion.category.value}")
            lines.append(f"   {suggestion.description}")
            
            if suggestion.related_failures:
                lines.append(f"   Related failures: {', '.join(suggestion.related_failures)}")
            
            if suggestion.implementation_hints:
                lines.append("   How to fix:")
                for hint in suggestion.implementation_hints[:3]:
                    lines.append(f"     • {hint}")
            
            if suggestion.expected_impact:
                lines.append(f"   Expected impact: {suggestion.expected_impact}")
        
        return "\n".join(lines)
    
    def _format_html(self, suggestions: List[Suggestion]) -> str:
        """Format suggestions as HTML."""
        if not suggestions:
            return '<div class="suggestions"><p style="color: #4CAF50;">✅ No improvement suggestions - MAS is performing well!</p></div>'
        
        priority_colors = {
            Priority.CRITICAL: "#dc3545",
            Priority.HIGH: "#fd7e14",
            Priority.MEDIUM: "#ffc107",
            Priority.LOW: "#28a745"
        }
        
        html_parts = ['<div class="suggestions">']
        
        for suggestion in suggestions:
            color = priority_colors[suggestion.priority]
            html_parts.append(f'''
            <div class="suggestion" style="border-left: 4px solid {color}; padding: 12px; margin: 10px 0; background: #f8f9fa; border-radius: 4px;">
                <h4 style="margin: 0 0 8px 0; color: {color};">
                    [{suggestion.priority.value.upper()}] {suggestion.title}
                </h4>
                <p style="margin: 0 0 8px 0;">{suggestion.description}</p>
                <small style="color: #666;">Category: {suggestion.category.value}</small>
            ''')
            
            if suggestion.implementation_hints:
                html_parts.append('<ul style="margin: 8px 0; padding-left: 20px;">')
                for hint in suggestion.implementation_hints[:3]:
                    html_parts.append(f'<li>{hint}</li>')
                html_parts.append('</ul>')
            
            if suggestion.expected_impact:
                html_parts.append(f'<p style="color: #28a745; margin: 8px 0 0 0;"><strong>Expected impact:</strong> {suggestion.expected_impact}</p>')
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def get_summary(self, suggestions: List[Suggestion]) -> Dict[str, Any]:
        """Get summary statistics of suggestions."""
        if not suggestions:
            return {
                "total": 0,
                "by_priority": {},
                "by_category": {},
                "top_issues": []
            }
        
        by_priority = {}
        by_category = {}
        
        for s in suggestions:
            by_priority[s.priority.value] = by_priority.get(s.priority.value, 0) + 1
            by_category[s.category.value] = by_category.get(s.category.value, 0) + 1
        
        return {
            "total": len(suggestions),
            "by_priority": by_priority,
            "by_category": by_category,
            "top_issues": [s.title for s in suggestions[:3]]
        }
