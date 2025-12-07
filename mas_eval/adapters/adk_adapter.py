"""
Google ADK Adapter for MAS Evaluation Framework.

Provides integration with Google Agent Development Kit (ADK) to capture
agent execution traces for evaluation.

Enhanced with:
- Embedding-based thought/output classification
- Proper parent span tracking
- Actual event timestamps
- Error span capture
- Content truncation markers
- Max span limits for large traces
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import asyncio
import warnings

from mas_eval.core.types import Span, StepType, TraceData


# ===== Text Classification Configuration =====
class TextClassifier:
    """
    Semantic classifier for distinguishing thoughts from outputs.
    Uses sentence embeddings instead of pattern matching.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._encoder = None
        self._thought_prototypes = None
        self._output_prototypes = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Lazy initialize the encoder and prototypes."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self._encoder = SentenceTransformer(self.model_name)
            
            # Create prototype embeddings for thoughts and outputs
            thought_examples = [
                "Let me think about this step by step",
                "I need to analyze the data first",
                "My plan is to approach this by",
                "Considering the options available",
                "I should investigate further",
                "Breaking down the problem into parts",
                "First, I will research the topic",
                "Analyzing the requirements carefully",
                "I believe the best approach would be",
                "Thinking through the implications",
            ]
            
            output_examples = [
                "Here is the final report",
                "The results show that",
                "In conclusion, the analysis reveals",
                "Based on my research, here are the findings",
                "The answer to your question is",
                "Summary of the completed work",
                "Final recommendation:",
                "Here are my findings:",
                "The completed analysis shows",
                "To summarize the key points:",
            ]
            
            thought_embeddings = self._encoder.encode(thought_examples)
            output_embeddings = self._encoder.encode(output_examples)
            
            # Store centroids as prototypes
            self._thought_prototypes = np.mean(thought_embeddings, axis=0)
            self._output_prototypes = np.mean(output_embeddings, axis=0)
            
            self._initialized = True
            
        except ImportError:
            warnings.warn(
                "sentence-transformers not available. "
                "Falling back to pattern-based classification."
            )
            self._initialized = False
    
    def classify(self, text: str) -> Tuple[StepType, float]:
        """
        Classify text as THOUGHT or OUTPUT with confidence score.
        
        Args:
            text: The text to classify
            
        Returns:
            Tuple of (StepType, confidence_score)
        """
        if not self._initialized:
            self._initialize()
        
        if self._encoder is None:
            # Fallback to pattern matching
            return self._pattern_classify(text)
        
        try:
            import numpy as np
            from numpy.linalg import norm
            
            # Get embedding for input text
            embedding = self._encoder.encode([text])[0]
            
            # Calculate cosine similarity to prototypes
            thought_sim = np.dot(embedding, self._thought_prototypes) / (
                norm(embedding) * norm(self._thought_prototypes)
            )
            output_sim = np.dot(embedding, self._output_prototypes) / (
                norm(embedding) * norm(self._output_prototypes)
            )
            
            # Determine classification
            if thought_sim > output_sim:
                confidence = (thought_sim - output_sim + 1) / 2  # Normalize to [0, 1]
                return StepType.THOUGHT, min(0.99, confidence)
            else:
                confidence = (output_sim - thought_sim + 1) / 2
                return StepType.OUTPUT, min(0.99, confidence)
                
        except Exception as e:
            warnings.warn(f"Embedding classification failed: {e}. Using fallback.")
            return self._pattern_classify(text)
    
    def _pattern_classify(self, text: str) -> Tuple[StepType, float]:
        """Fallback pattern-based classification."""
        text_lower = text.lower()
        
        thought_patterns = [
            'let me', 'i will', 'i need to', 'i should', "i'll",
            'thinking', 'analyzing', 'considering', 'planning',
            'first,', 'next,', 'then,', 'step', 'my plan',
            'based on', 'reviewing', 'examining', 'i think', 'i believe'
        ]
        
        output_patterns = [
            'here is', 'here are', 'the result', 'the answer',
            'in conclusion', 'summary:', 'final', 'findings:',
            'recommendation:', 'report:', 'completed'
        ]
        
        thought_score = sum(1 for p in thought_patterns if p in text_lower)
        output_score = sum(1 for p in output_patterns if p in text_lower)
        
        if thought_score > output_score:
            confidence = min(0.8, 0.5 + thought_score * 0.1)
            return StepType.THOUGHT, confidence
        elif output_score > 0:
            confidence = min(0.8, 0.5 + output_score * 0.1)
            return StepType.OUTPUT, confidence
        else:
            # Default to thought with low confidence
            return StepType.THOUGHT, 0.5


# Global classifier instance (lazy loaded)
_text_classifier: Optional[TextClassifier] = None

def get_text_classifier() -> TextClassifier:
    """Get or create the global text classifier."""
    global _text_classifier
    if _text_classifier is None:
        _text_classifier = TextClassifier()
    return _text_classifier


class ADKTracingCallback:
    """
    Callback handler that captures ADK agent events as Spans.
    
    Enhanced Features:
    - Embedding-based thought/output classification
    - Proper parent span validation
    - Max span limits for large traces
    - Error span capture
    - Content truncation with markers
    
    Usage:
        callback = ADKTracingCallback(max_spans=1000)
        
        # Run your ADK agents with callback
        async for event in runner.run_async(...):
            callback.on_event(event)
        
        # Get captured spans
        spans = callback.get_spans()
    """
    
    # Default max spans to prevent memory issues
    DEFAULT_MAX_SPANS = 5000
    
    def __init__(
        self, 
        service_name: str = "adk-mas",
        max_spans: int = None,
        verbose: bool = True,
        use_embedding_classifier: bool = True
    ):
        """
        Initialize the ADK tracing callback.
        
        Args:
            service_name: Name for the tracing service
            max_spans: Maximum number of spans to store (None = no limit)
            verbose: Print trace events to console
            use_embedding_classifier: Use embeddings for thought classification
        """
        self.service_name = service_name
        self.max_spans = max_spans or self.DEFAULT_MAX_SPANS
        self.verbose = verbose
        self.use_embedding_classifier = use_embedding_classifier
        
        self._spans: List[Span] = []
        self._span_stack: List[str] = []  # Global stack for hierarchical relationships
        self._agent_span_stack: Dict[str, List[str]] = {}  # Per-agent span tracking
        self._last_agent_span: Dict[str, str] = {}  # Last span per agent for temporal linking
        self._agent_contexts: Dict[str, str] = {}  # Active context per agent
        self._span_id_set: set = set()  # Fast lookup for span existence
        
        # GEMMAS: Track iteration count per agent for (Agent_ID, Iteration, Role) nodes
        self._agent_iterations: Dict[str, int] = {}
        
        self._trace_id = str(uuid.uuid4())
        self._counter = 0
        self._current_agent: Optional[str] = None
        self._evicted_count = 0  # Track how many spans were evicted
        
        # Initialize text classifier if enabled
        self._classifier: Optional[TextClassifier] = None
        if use_embedding_classifier:
            self._classifier = get_text_classifier()
    
    def _generate_span_id(self) -> str:
        self._counter += 1
        span_id = f"span_{self._counter:04d}"
        self._span_id_set.add(span_id)
        return span_id
    
    def _get_parent_span_id(self, agent_name: str = None) -> Optional[str]:
        """
        Get parent span ID with proper validation.
        
        Priority:
        1. Agent-specific context (from transfers)
        2. Agent's own last span (temporal)
        3. Global stack (delegation)
        4. None (no parent - this is OK for root spans)
        
        Note: We no longer fall back to arbitrary last spans, which was
        causing incorrect causal relationships in the GOT.
        """
        # First try agent's active context (set during transfers)
        if agent_name and agent_name in self._agent_contexts:
            ctx = self._agent_contexts[agent_name]
            if ctx in self._span_id_set:
                return ctx
        
        # Try agent-specific stack
        if agent_name and agent_name in self._agent_span_stack:
            agent_stack = self._agent_span_stack[agent_name]
            if agent_stack:
                last_span = agent_stack[-1]
                if last_span in self._span_id_set:
                    return last_span
        
        # Try the agent's last known span (temporal parent for same agent)
        if agent_name and agent_name in self._last_agent_span:
            last_span = self._last_agent_span[agent_name]
            if last_span in self._span_id_set:
                return last_span
        
        # Try global stack (for delegation chains)
        if self._span_stack:
            for span_id in reversed(self._span_stack):
                if span_id in self._span_id_set:
                    return span_id
        
        # No parent found - this is fine for root spans
        # DO NOT fall back to arbitrary last span - this causes incorrect causality
        return None
    
    def _validate_parent_exists(self, parent_id: Optional[str]) -> Optional[str]:
        """Ensure the parent span actually exists."""
        if parent_id is None:
            return None
        if parent_id in self._span_id_set:
            return parent_id
        return None
    
    def _get_agent_iteration(self, agent_name: str) -> int:
        """Get current iteration counter for an agent (GEMMAS requirement)."""
        return self._agent_iterations.get(agent_name, 0)
    
    def _increment_agent_iteration(self, agent_name: str) -> int:
        """Increment and return iteration counter for an agent."""
        if agent_name not in self._agent_iterations:
            self._agent_iterations[agent_name] = 0
        self._agent_iterations[agent_name] += 1
        return self._agent_iterations[agent_name]
    
    def _register_span(self, agent_name: str, span_id: str) -> None:
        """Register a span for an agent and increment iteration counter."""
        self._last_agent_span[agent_name] = span_id
        if agent_name not in self._agent_span_stack:
            self._agent_span_stack[agent_name] = []
        # Increment iteration when registering a new span for this agent
        self._increment_agent_iteration(agent_name)
    
    def _map_event_type(self, event_type: str) -> StepType:
        """Map ADK event types to our StepType."""
        mapping = {
            "thought": StepType.THOUGHT,
            "action": StepType.ACTION,
            "observation": StepType.OBSERVATION,
            "output": StepType.OUTPUT,
            "tool_call": StepType.TOOL_CALL,
            "tool_result": StepType.TOOL_RESULT,
            "message": StepType.MESSAGE,
            "error": StepType.ERROR,
            # ADK specific mappings
            "model_turn": StepType.THOUGHT,
            "function_call": StepType.TOOL_CALL,
            "function_response": StepType.TOOL_RESULT,
            "agent_transfer": StepType.ACTION,
        }
        return mapping.get(event_type.lower(), StepType.ACTION)
    
    def on_event(self, event: Any) -> None:
        """
        Process an ADK event and create Span(s).
        
        ADK Event structure:
        - id: unique event identifier
        - author: 'user' or agent name
        - content: contains parts with text, function_call, function_response
        - get_function_calls(): list of function calls
        - get_function_responses(): list of function responses
        - actions: EventActions with state_delta, transfer_to_agent, etc.
        
        Args:
            event: ADK event object (from runner.run_async)
        """
        try:
            # Get agent/author name
            agent_name = getattr(event, 'author', None) or 'Unknown'
            if agent_name == 'user':
                # Skip user input events, we only want agent events
                return
            
            # Debug logging - print event info to help diagnose
            event_type_name = type(event).__name__
            print(f"[TRACE] Event from '{agent_name}' ({event_type_name})")
            
            # Process function calls if present
            if hasattr(event, 'get_function_calls'):
                func_calls = event.get_function_calls()
                if func_calls:
                    for fc in func_calls:
                        self._create_function_call_span(agent_name, fc)
            
            # Process function responses if present
            if hasattr(event, 'get_function_responses'):
                func_responses = event.get_function_responses()
                if func_responses:
                    for fr in func_responses:
                        self._create_function_response_span(agent_name, fr)
            
            # Process text content - this includes thoughts and outputs
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text = part.text.strip()
                            if text:
                                self._create_text_span(agent_name, text)
                        # Also check for inline function calls/responses in parts
                        elif hasattr(part, 'function_call') and part.function_call:
                            self._create_function_call_span(agent_name, part.function_call)
                        elif hasattr(part, 'function_response') and part.function_response:
                            self._create_function_response_span(agent_name, part.function_response)
            
            # Process agent transfers (actions)
            if hasattr(event, 'actions') and event.actions:
                actions = event.actions
                if hasattr(actions, 'transfer_to_agent') and actions.transfer_to_agent:
                    self._create_transfer_span(agent_name, actions.transfer_to_agent)
                    
        except Exception as e:
            print(f"Warning: Error processing ADK event: {e}")
            import traceback
            traceback.print_exc()
    
    def _truncate_with_marker(self, text: str, max_len: int = 2000) -> str:
        """Truncate content with marker if needed."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 20] + "\n... [TRUNCATED]"
    
    def _add_span(self, span: Span) -> None:
        """Add a span with limit enforcement."""
        if len(self._spans) >= self.max_spans:
            # Evict oldest spans (keep at least 10% for context)
            evict_count = max(1, self.max_spans // 10)
            self._spans = self._spans[evict_count:]
            self._evicted_count += evict_count
            if self.verbose:
                print(f"[TRACE] Warning: Evicted {evict_count} spans (limit: {self.max_spans})")
        self._spans.append(span)
    
    def _extract_event_timestamp(self, event: Any) -> datetime:
        """Extract actual timestamp from ADK event if available."""
        # Try various timestamp attributes from ADK event
        for attr in ['timestamp', 'created_at', 'time', '_timestamp']:
            ts = getattr(event, attr, None)
            if ts is not None:
                if isinstance(ts, datetime):
                    return ts
                elif isinstance(ts, (int, float)):
                    # Assume epoch seconds or milliseconds
                    if ts > 1e12:  # milliseconds
                        return datetime.fromtimestamp(ts / 1000)
                    return datetime.fromtimestamp(ts)
        return datetime.now()

    def _create_text_span(self, agent_name: str, text: str, event: Any = None) -> None:
        """Create a span from text content using semantic classification.
        
        Uses embedding-based classification for accurate thought/output detection.
        Falls back to pattern matching if embeddings unavailable.
        """
        # Use semantic classifier if available
        if self._classifier is not None:
            step_type, confidence = self._classifier.classify(text)
            event_type = "thought" if step_type == StepType.THOUGHT else "output"
        else:
            # Fallback to simple pattern matching
            text_lower = text.lower()
            
            thought_indicators = ['let me', 'i will', 'i need', 'thinking', 'analyzing', 
                                  'first,', 'step', 'plan', 'based on', 'considering']
            output_indicators = ['here is', 'result', 'conclusion', 'summary', 'final']
            
            thought_score = sum(1 for p in thought_indicators if p in text_lower)
            output_score = sum(1 for p in output_indicators if p in text_lower)
            
            if output_score > thought_score:
                step_type = StepType.OUTPUT
                event_type = "output"
                confidence = 0.6
            else:
                step_type = StepType.THOUGHT
                event_type = "thought"
                confidence = 0.5
        
        # Extract timestamp from event if available
        timestamp = self._extract_event_timestamp(event) if event else datetime.now()
        
        span_id = self._generate_span_id()
        current_iteration = self._get_agent_iteration(agent_name)
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(agent_name),
            name=f"{agent_name}.{event_type}",
            agent_name=str(agent_name),
            step_type=step_type,
            content=self._truncate_with_marker(text),
            start_time=timestamp,
            iteration=current_iteration,  # GEMMAS: Track iteration
            attributes={
                "trace_id": self._trace_id,
                "event_category": "text",
                "classification_confidence": confidence,
                "classifier_type": "embedding" if self._classifier else "pattern"
            }
        )
        self._add_span(span)
        self._register_span(agent_name, span_id)
        if self.verbose:
            conf_str = f" (conf: {confidence:.2f})" if self._classifier else ""
            print(f"[TRACE]   -> Created {event_type}{conf_str} span: {text[:80]}...")
    
    def _create_function_call_span(self, agent_name: str, func_call: Any) -> None:
        """Create a span from a function call."""
        try:
            func_name = getattr(func_call, 'name', 'unknown_function')
            func_args = getattr(func_call, 'args', {})
            if hasattr(func_args, 'items'):
                args_str = str(dict(func_args))
            else:
                args_str = str(func_args)
            
            content = f"Calling {func_name}({args_str})"
            
            span_id = self._generate_span_id()
            span = Span(
                span_id=span_id,
                parent_span_id=self._get_parent_span_id(agent_name),
                name=f"{agent_name}.tool_call",
                agent_name=str(agent_name),
                step_type=StepType.TOOL_CALL,
                content=self._truncate_with_marker(content),
                start_time=datetime.now(),
                attributes={
                    "trace_id": self._trace_id,
                    "event_category": "function_call",
                    "function_name": func_name
                }
            )
            self._add_span(span)
            self._register_span(agent_name, span_id)
            if self.verbose:
                print(f"[TRACE]   -> Created tool_call span: {func_name}")
        except Exception as e:
            self._create_error_span(agent_name, e, "function_call")
    
    def _create_function_response_span(self, agent_name: str, func_response: Any) -> None:
        """Create a span from a function response."""
        try:
            func_name = getattr(func_response, 'name', 'unknown_function')
            response_data = getattr(func_response, 'response', {})
            
            content = f"Result from {func_name}: {str(response_data)[:500]}"
            
            span_id = self._generate_span_id()
            span = Span(
                span_id=span_id,
                parent_span_id=self._get_parent_span_id(agent_name),
                name=f"{agent_name}.tool_result",
                agent_name=str(agent_name),
                step_type=StepType.TOOL_RESULT,
                content=self._truncate_with_marker(content),
                start_time=datetime.now(),
                attributes={
                    "trace_id": self._trace_id,
                    "event_category": "function_response",
                    "function_name": func_name
                }
            )
            self._add_span(span)
            self._register_span(agent_name, span_id)
            if self.verbose:
                print(f"[TRACE]   -> Created tool_result span: {func_name}")
        except Exception as e:
            self._create_error_span(agent_name, e, "function_response")
    
    def _create_transfer_span(self, agent_name: str, target_agent: str) -> None:
        """Create a span for agent transfer action (tracks Spatial Matrix S)."""
        span_id = self._generate_span_id()
        current_iteration = self._get_agent_iteration(agent_name)
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(agent_name),
            name=f"{agent_name}.transfer",
            agent_name=str(agent_name),
            step_type=StepType.ACTION,
            content=f"Transferring to agent: {target_agent}",
            start_time=datetime.now(),
            iteration=current_iteration,  # GEMMAS: Track iteration
            from_agent=agent_name,  # GEMMAS: Source for Spatial Matrix
            to_agent=target_agent,   # GEMMAS: Target for Spatial Matrix
            attributes={
                "trace_id": self._trace_id,
                "event_category": "transfer",
                "target_agent": target_agent,
                "from_agent": agent_name,
                "to_agent": target_agent
            }
        )
        self._add_span(span)
        self._register_span(agent_name, span_id)
        # Set global stack for sub-agent to inherit parent context
        self._span_stack.append(span_id)
        # Set context for target agent
        self._agent_contexts[target_agent] = span_id
        if self.verbose:
            print(f"[TRACE]   -> Created transfer span: {agent_name} -> {target_agent}")
    
    def _create_error_span(self, agent_name: str, error: Exception, context: str = "") -> None:
        """Create a span for captured errors."""
        span_id = self._generate_span_id()
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(agent_name),
            name=f"{agent_name}.error",
            agent_name=str(agent_name),
            step_type=StepType.ERROR,
            content=f"Error in {context}: {type(error).__name__}: {str(error)}",
            start_time=datetime.now(),
            attributes={
                "trace_id": self._trace_id,
                "event_category": "error",
                "error_type": type(error).__name__,
                "error_context": context
            }
        )
        self._add_span(span)
        self._register_span(agent_name, span_id)
        if self.verbose:
            print(f"[TRACE]   -> Created error span: {type(error).__name__} in {context}")
    
    def on_agent_start(self, agent_name: str, instruction: str = "") -> str:
        """Record agent start event."""
        span_id = self._generate_span_id()
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(),
            name=f"{agent_name}.start",
            agent_name=agent_name,
            step_type=StepType.ACTION,
            content=f"Agent started: {instruction[:500]}",
            start_time=datetime.now(),
            attributes={"trace_id": self._trace_id}
        )
        self._spans.append(span)
        self._span_stack.append(span_id)
        return span_id
    
    def on_agent_end(self, agent_name: str, result: str = "") -> None:
        """Record agent end event."""
        span_id = self._generate_span_id()
        parent_id = self._span_stack.pop() if self._span_stack else None
        span = Span(
            span_id=span_id,
            parent_span_id=parent_id,
            name=f"{agent_name}.end",
            agent_name=agent_name,
            step_type=StepType.OUTPUT,
            content=f"Agent completed: {result[:500]}",
            start_time=datetime.now(),
            attributes={"trace_id": self._trace_id}
        )
        self._spans.append(span)
    
    def on_thought(self, agent_name: str, thought: str) -> None:
        """Record agent thought."""
        span_id = self._generate_span_id()
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(),
            name=f"{agent_name}.thought",
            agent_name=agent_name,
            step_type=StepType.THOUGHT,
            content=thought[:2000],
            start_time=datetime.now(),
            attributes={"trace_id": self._trace_id}
        )
        self._spans.append(span)
    
    def on_action(self, agent_name: str, action: str) -> None:
        """Record agent action."""
        span_id = self._generate_span_id()
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(),
            name=f"{agent_name}.action",
            agent_name=agent_name,
            step_type=StepType.ACTION,
            content=action[:2000],
            start_time=datetime.now(),
            attributes={"trace_id": self._trace_id}
        )
        self._spans.append(span)
    
    def on_output(self, agent_name: str, output: str) -> None:
        """Record agent output."""
        span_id = self._generate_span_id()
        span = Span(
            span_id=span_id,
            parent_span_id=self._get_parent_span_id(),
            name=f"{agent_name}.output",
            agent_name=agent_name,
            step_type=StepType.OUTPUT,
            content=output[:2000],
            start_time=datetime.now(),
            attributes={"trace_id": self._trace_id}
        )
        self._spans.append(span)
    
    def get_spans(self) -> List[Span]:
        """Get all collected spans."""
        return self._spans.copy()
    
    def get_trace_data(self) -> TraceData:
        """Get complete trace data."""
        return TraceData(
            trace_id=self._trace_id,
            spans=self.get_spans(),
            metadata={"service_name": self.service_name}
        )
    
    def clear(self) -> None:
        """Clear all collected spans."""
        self._spans.clear()
        self._span_stack.clear()
        self._trace_id = str(uuid.uuid4())
        self._counter = 0


class ADKAdapter:
    """
    High-level adapter for running ADK agents with tracing.
    
    Wraps the ADK Runner and automatically captures all events as Spans.
    
    Usage:
        from google.adk.agents import Agent
        from google.adk.runners import InMemoryRunner
        
        agent = Agent(name="MyAgent", model="gemini-2.5-flash", ...)
        adapter = ADKAdapter(agent)
        
        # Run with tracing
        result, spans = await adapter.run_with_tracing("Hello, what can you do?")
        
        # Use spans for evaluation
        evaluator.evaluate(spans)
    """
    
    def __init__(
        self, 
        agent: Any,
        session_service: Optional[Any] = None,
        artifact_service: Optional[Any] = None,
        service_name: str = "adk-mas"
    ):
        """
        Initialize the ADK adapter.
        
        Args:
            agent: The root ADK Agent instance
            session_service: Optional session service (InMemorySessionService if None)
            artifact_service: Optional artifact service (InMemoryArtifactService if None)
            service_name: Name for the tracing service
        """
        self.agent = agent
        self.service_name = service_name
        self._runner = None
        self._session_service = session_service
        self._artifact_service = artifact_service
        self._callback = ADKTracingCallback(service_name=service_name)
    
    def _initialize_runner(self):
        """Lazily initialize the ADK runner."""
        if self._runner is None:
            try:
                from google.adk.runners import InMemoryRunner
                
                # Note: The new InMemoryRunner API handles session/artifact services internally
                self._runner = InMemoryRunner(
                    agent=self.agent,
                    app_name=self.service_name
                )
            except ImportError:
                raise ImportError(
                    "google-adk required. Install with: pip install google-adk"
                )
    
    async def run_with_tracing(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: str = "user"
    ) -> tuple[str, List[Span]]:
        """
        Run the agent with full tracing.
        
        Args:
            user_message: The user's input message
            session_id: Optional session ID (generated if not provided)
            user_id: User identifier
            
        Returns:
            Tuple of (final_response, spans)
        """
        self._initialize_runner()
        self._callback.clear()
        
        session_id = session_id or str(uuid.uuid4())
        final_response = ""
        
        try:
            from google.genai import types
            
            # Create session first - required by the new ADK API
            session = await self._runner.session_service.create_session(
                app_name=self.service_name,
                user_id=user_id,
                session_id=session_id
            )
            
            content = types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            )
            
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=content
            ):
                self._callback.on_event(event)
                
                # Capture final response
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_response = part.text
                                
        except Exception as e:
            self._callback.on_action("System", f"Error during execution: {str(e)}")
            raise
        
        return final_response, self._callback.get_spans()
    
    def run_with_tracing_sync(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: str = "user"
    ) -> tuple[str, List[Span]]:
        """
        Synchronous version of run_with_tracing.
        
        Args:
            user_message: The user's input message
            session_id: Optional session ID
            user_id: User identifier
            
        Returns:
            Tuple of (final_response, spans)
        """
        return asyncio.run(
            self.run_with_tracing(user_message, session_id, user_id)
        )
    
    def get_callback(self) -> ADKTracingCallback:
        """Get the tracing callback for manual event recording."""
        return self._callback
    
    def get_trace_data(self) -> TraceData:
        """Get complete trace data from the last run."""
        return self._callback.get_trace_data()


def create_adk_adapter(agent: Any, **kwargs) -> ADKAdapter:
    """Factory function to create an ADK adapter."""
    return ADKAdapter(agent, **kwargs)
