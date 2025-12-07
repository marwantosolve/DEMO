"""
MAST Taxonomy - Multi-Agent System Failure Taxonomy.

Defines the 14 failure modes from the "Why Do Multi-Agent LLM Systems Fail?" paper,
organized into three categories:
1. Specification Issues (5 modes)
2. Inter-Agent Misalignment (6 modes)
3. Task Verification (3 modes)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class FailureCategory(Enum):
    """Main categories of MAST failure modes."""
    SPECIFICATION = "specification"
    INTER_AGENT = "inter_agent"
    TASK_VERIFICATION = "task_verification"


@dataclass
class FailureModeDefinition:
    """Definition of a single MAST failure mode."""
    code: str
    name: str
    category: FailureCategory
    description: str
    indicators: List[str]
    severity: str  # "high", "medium", "low"
    frequency: float  # Observed frequency from paper (%)
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "indicators": self.indicators,
            "severity": self.severity,
            "frequency": self.frequency
        }


class MASTTaxonomy:
    """
    Complete MAST Taxonomy with all 14 failure modes.
    
    Based on: "Why Do Multi-Agent LLM Systems Fail?" research paper.
    
    Usage:
        taxonomy = MASTTaxonomy()
        
        # Get all failure modes
        all_modes = taxonomy.get_all_modes()
        
        # Get by category
        inter_agent_modes = taxonomy.get_by_category(FailureCategory.INTER_AGENT)
        
        # Get specific mode
        mode = taxonomy.get_mode("INTER-3")
    """
    
    # ===== SPECIFICATION ISSUES (5 modes) =====
    SPEC_1 = FailureModeDefinition(
        code="SPEC-1",
        name="Disobey Task Specification",
        category=FailureCategory.SPECIFICATION,
        description="Agent fails to adhere to constraints or requirements explicitly stated in the task specification.",
        indicators=[
            "Output contradicts explicit task requirements",
            "Agent ignores format specifications",
            "Required steps are skipped",
            "Output doesn't match expected type/structure"
        ],
        severity="high",
        frequency=12.5
    )
    
    SPEC_2 = FailureModeDefinition(
        code="SPEC-2",
        name="Disobey Role Specification",
        category=FailureCategory.SPECIFICATION,
        description="Agent deviates from their assigned role or responsibilities within the multi-agent system.",
        indicators=[
            "Agent performs tasks outside their defined role",
            "Role boundaries are violated",
            "Agent assumes responsibilities of another agent",
            "Role-specific constraints ignored"
        ],
        severity="high",
        frequency=8.3
    )
    
    SPEC_3 = FailureModeDefinition(
        code="SPEC-3",
        name="Step Repetition",
        category=FailureCategory.SPECIFICATION,
        description="System unnecessarily reiterates previously completed steps, leading to inefficiency and potential loops.",
        indicators=[
            "Same action performed multiple times",
            "Circular conversation patterns",
            "Redundant processing of same data",
            "Agent re-asks answered questions"
        ],
        severity="medium",
        frequency=15.2
    )
    
    SPEC_4 = FailureModeDefinition(
        code="SPEC-4",
        name="Loss of Conversation History",
        category=FailureCategory.SPECIFICATION,
        description="Context of ongoing dialogue is unexpectedly truncated or lost, impacting coherent interactions.",
        indicators=[
            "Agent forgets previous context",
            "Earlier decisions not referenced",
            "Information needs to be repeated",
            "Inconsistent with previous statements"
        ],
        severity="high",
        frequency=6.8
    )
    
    SPEC_5 = FailureModeDefinition(
        code="SPEC-5",
        name="Unaware of Termination Conditions",
        category=FailureCategory.SPECIFICATION,
        description="System fails to recognize when a task or sub-task should conclude.",
        indicators=[
            "Task continues after completion criteria met",
            "Endless refinement loops",
            "No clear stopping point",
            "Agent doesn't signal completion"
        ],
        severity="medium",
        frequency=5.4
    )
    
    # ===== INTER-AGENT MISALIGNMENT (6 modes) =====
    INTER_1 = FailureModeDefinition(
        code="INTER-1",
        name="Conversation Reset",
        category=FailureCategory.INTER_AGENT,
        description="Agent unexpectedly restarts dialogue, causing loss of accumulated context and progress.",
        indicators=[
            "Sudden topic change without transition",
            "Previous agreements forgotten",
            "Conversation starts from scratch",
            "Context appears to be wiped"
        ],
        severity="high",
        frequency=2.33
    )
    
    INTER_2 = FailureModeDefinition(
        code="INTER-2",
        name="Fail to Ask for Clarification",
        category=FailureCategory.INTER_AGENT,
        description="Agents do not request necessary information when confronted with ambiguity or uncertainty.",
        indicators=[
            "Proceeds with incomplete information",
            "Assumes meaning without verification",
            "No follow-up questions on ambiguous input",
            "Makes decisions without sufficient data"
        ],
        severity="medium",
        frequency=11.65
    )
    
    INTER_3 = FailureModeDefinition(
        code="INTER-3",
        name="Task Derailment",
        category=FailureCategory.INTER_AGENT,
        description="Multi-agent system deviates from the intended objective or goal.",
        indicators=[
            "Discussion moves away from main task",
            "Focus shifts to tangential topics",
            "Original objective not addressed",
            "Output doesn't solve stated problem"
        ],
        severity="high",
        frequency=7.15
    )
    
    INTER_4 = FailureModeDefinition(
        code="INTER-4",
        name="Information Withholding",
        category=FailureCategory.INTER_AGENT,
        description="Agent fails to share important data or insights with other agents critical for task completion.",
        indicators=[
            "Relevant information not passed to downstream agents",
            "Selective sharing of results",
            "Key findings hidden or omitted",
            "Incomplete handoff between agents"
        ],
        severity="high",
        frequency=1.66
    )
    
    INTER_5 = FailureModeDefinition(
        code="INTER-5",
        name="Ignored Other Agent's Output",
        category=FailureCategory.INTER_AGENT,
        description="Agent neglects or overlooks relevant information provided by another agent.",
        indicators=[
            "Previous agent's output not referenced",
            "Contradicts earlier agent's findings",
            "Doesn't build on prior work",
            "Duplicates completed work"
        ],
        severity="high",
        frequency=0.17
    )
    
    INTER_6 = FailureModeDefinition(
        code="INTER-6",
        name="Reasoning-Action Mismatch",
        category=FailureCategory.INTER_AGENT,
        description="Discrepancy between an agent's reasoning process and its subsequent actions.",
        indicators=[
            "Stated plan differs from execution",
            "Reasoning leads to different conclusion than action",
            "Inconsistency between 'thinking' and 'doing'",
            "Action contradicts stated logic"
        ],
        severity="high",
        frequency=13.98
    )
    
    # ===== TASK VERIFICATION (3 modes) =====
    TASK_1 = FailureModeDefinition(
        code="TASK-1",
        name="Premature Termination",
        category=FailureCategory.TASK_VERIFICATION,
        description="System stops execution before the task is genuinely completed.",
        indicators=[
            "Task declared complete but requirements not met",
            "Early exit without full result",
            "Partial output presented as final",
            "Missing expected components in output"
        ],
        severity="high",
        frequency=8.7
    )
    
    TASK_2 = FailureModeDefinition(
        code="TASK-2",
        name="Fail to Identify Task Completion",
        category=FailureCategory.TASK_VERIFICATION,
        description="System incorrectly believes a task is finished when it has not been, or fails to recognize successful completion.",
        indicators=[
            "Continues working after task is done",
            "Doesn't acknowledge success",
            "Keeps refining completed work",
            "No clear completion signal"
        ],
        severity="medium",
        frequency=4.2
    )
    
    TASK_3 = FailureModeDefinition(
        code="TASK-3",
        name="Fail to Identify Task Failure",
        category=FailureCategory.TASK_VERIFICATION,
        description="System does not detect or acknowledge that a task has failed.",
        indicators=[
            "Error not reported",
            "Incorrect result presented as success",
            "Failure conditions ignored",
            "No error handling for obvious failures"
        ],
        severity="high",
        frequency=3.8
    )
    
    def __init__(self):
        """Initialize taxonomy with all failure modes."""
        self._modes: Dict[str, FailureModeDefinition] = {
            # Specification Issues
            "SPEC-1": self.SPEC_1,
            "SPEC-2": self.SPEC_2,
            "SPEC-3": self.SPEC_3,
            "SPEC-4": self.SPEC_4,
            "SPEC-5": self.SPEC_5,
            # Inter-Agent Misalignment
            "INTER-1": self.INTER_1,
            "INTER-2": self.INTER_2,
            "INTER-3": self.INTER_3,
            "INTER-4": self.INTER_4,
            "INTER-5": self.INTER_5,
            "INTER-6": self.INTER_6,
            # Task Verification
            "TASK-1": self.TASK_1,
            "TASK-2": self.TASK_2,
            "TASK-3": self.TASK_3
        }
    
    def get_all_modes(self) -> List[FailureModeDefinition]:
        """Get all 14 failure modes."""
        return list(self._modes.values())
    
    def get_mode(self, code: str) -> Optional[FailureModeDefinition]:
        """Get a specific failure mode by code."""
        return self._modes.get(code)
    
    def get_by_category(self, category: FailureCategory) -> List[FailureModeDefinition]:
        """Get all failure modes in a category."""
        return [m for m in self._modes.values() if m.category == category]
    
    def get_specification_modes(self) -> List[FailureModeDefinition]:
        """Get Specification Issues failure modes."""
        return self.get_by_category(FailureCategory.SPECIFICATION)
    
    def get_inter_agent_modes(self) -> List[FailureModeDefinition]:
        """Get Inter-Agent Misalignment failure modes."""
        return self.get_by_category(FailureCategory.INTER_AGENT)
    
    def get_task_verification_modes(self) -> List[FailureModeDefinition]:
        """Get Task Verification failure modes."""
        return self.get_by_category(FailureCategory.TASK_VERIFICATION)
    
    def get_codes(self) -> List[str]:
        """Get all failure mode codes."""
        return list(self._modes.keys())
    
    def summary(self) -> str:
        """Generate summary of all failure modes."""
        lines = ["MAST Taxonomy - 14 Failure Modes", "=" * 50, ""]
        
        for category in FailureCategory:
            modes = self.get_by_category(category)
            lines.append(f"\n{category.value.upper().replace('_', ' ')} ({len(modes)} modes)")
            lines.append("-" * 40)
            for mode in modes:
                lines.append(f"  [{mode.code}] {mode.name}")
                lines.append(f"    Severity: {mode.severity} | Frequency: {mode.frequency}%")
        
        return "\n".join(lines)
    
    def to_prompt_context(self) -> str:
        """Generate prompt context for LLM classification."""
        lines = [
            "MAST FAILURE TAXONOMY",
            "You must classify traces into these 14 failure modes:",
            ""
        ]
        
        for mode in self.get_all_modes():
            lines.append(f"[{mode.code}] {mode.name}")
            lines.append(f"  Description: {mode.description}")
            lines.append(f"  Indicators: {', '.join(mode.indicators[:3])}")
            lines.append("")
        
        return "\n".join(lines)
