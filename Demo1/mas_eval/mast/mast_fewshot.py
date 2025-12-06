"""
MAST Few-Shot Examples Module.

Loads labeled examples from the MAD (Multi-Agent System Traces Dataset) 
for in-context learning with the MAST classifier.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MASTExample:
    """A labeled example from the MAD dataset."""
    trace_text: str
    failure_codes: List[str]
    failure_descriptions: List[str]
    mas_type: str
    benchmark: str
    llm_model: str


class MASTFewShotLoader:
    """
    Load and manage few-shot examples from the MAD dataset.
    
    The MAD dataset contains labeled MAS execution traces with MAST 
    failure mode annotations.
    
    Usage:
        loader = MASTFewShotLoader("path/to/mast_dataset")
        
        # Get examples for a specific failure mode
        examples = loader.get_examples_for_mode("INTER-3", max_examples=3)
        
        # Get diverse examples for general few-shot prompting
        examples = loader.get_diverse_examples(max_total=5)
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            dataset_path: Path to mast_dataset directory. 
                         If None, searches common locations.
        """
        self.dataset_path = self._find_dataset(dataset_path)
        self._examples: List[Dict] = []
        self._examples_by_mode: Dict[str, List[Dict]] = {}
        self._loaded = False
    
    def _find_dataset(self, dataset_path: Optional[str]) -> Optional[str]:
        """Find and validate the dataset path."""
        if dataset_path and os.path.exists(dataset_path):
            if self._validate_dataset(dataset_path):
                return dataset_path
            else:
                print(f"[MAST] Warning: Dataset at {dataset_path} is invalid or empty")
        
        # Common locations to search
        search_paths = [
            "./mast_dataset",
            "../mast_dataset",
            "../../mast_dataset",
            os.path.expanduser("~/mast_dataset"),
        ]
        
        # Check for environment variable
        if os.environ.get("MAST_DATASET_PATH"):
            search_paths.insert(0, os.environ["MAST_DATASET_PATH"])
        
        for path in search_paths:
            if os.path.exists(path) and self._validate_dataset(path):
                return path
        
        return None
    
    def _validate_dataset(self, path: str) -> bool:
        """
        Validate that the dataset path contains valid MAST data.
        
        Args:
            path: Path to the dataset directory
            
        Returns:
            True if valid dataset found, False otherwise
        """
        # Check for expected JSON files
        json_files = [
            "MAD_human_labelled_dataset.json",
            "MAD_full_dataset.json"
        ]
        
        for json_file in json_files:
            full_path = os.path.join(path, json_file)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Validate structure
                    if isinstance(data, list) and len(data) > 0:
                        # Check for expected fields in first entry
                        first_entry = data[0]
                        if isinstance(first_entry, dict):
                            # Should have either 'annotations', 'trace', or known fields
                            expected_fields = ['annotations', 'trace', 'mas_type', 
                                                'benchmark_name', 'failure_modes']
                            if any(field in first_entry for field in expected_fields):
                                return True
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[MAST] Warning: Error reading {json_file}: {e}")
                    continue
        
        return False
    
    def load(self) -> bool:
        """
        Load examples from the dataset.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if self._loaded:
            return True
        
        if not self.dataset_path:
            print("Warning: MAST dataset not found")
            return False
        
        # Try human-labeled dataset first (higher quality)
        human_labeled = os.path.join(self.dataset_path, "MAD_human_labelled_dataset.json")
        full_dataset = os.path.join(self.dataset_path, "MAD_full_dataset.json")
        
        try:
            target_file = human_labeled if os.path.exists(human_labeled) else full_dataset
            
            if not os.path.exists(target_file):
                print(f"Warning: No dataset file found in {self.dataset_path}")
                return False
            
            print(f"[MAST] Loading examples from {os.path.basename(target_file)}...")
            
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse the dataset
            self._examples = data if isinstance(data, list) else [data]
            
            # Index by failure mode
            self._index_by_failure_mode()
            
            print(f"[MAST] Loaded {len(self._examples)} examples")
            print(f"[MAST] Failure modes found: {list(self._examples_by_mode.keys())}")
            
            self._loaded = True
            return True
            
        except Exception as e:
            print(f"Warning: Error loading MAST dataset: {e}")
            return False
    
    def _index_by_failure_mode(self) -> None:
        """Index examples by their failure mode codes."""
        self._examples_by_mode = {}
        
        for example in self._examples:
            # Extract failure codes from the example
            failure_codes = self._extract_failure_codes(example)
            
            for code in failure_codes:
                if code not in self._examples_by_mode:
                    self._examples_by_mode[code] = []
                self._examples_by_mode[code].append(example)
    
    def _extract_failure_codes(self, example: Dict) -> List[str]:
        """Extract failure mode codes from an example.
        
        The MAD dataset uses 'annotations' with 'failure mode' text descriptions
        and 'annotator_X' boolean fields.
        """
        codes = []
        
        # Handle MAD dataset format with 'annotations' list
        if 'annotations' in example:
            annotations = example['annotations']
            if isinstance(annotations, list):
                for i, ann in enumerate(annotations):
                    if isinstance(ann, dict):
                        # Check if any annotator marked this as a failure
                        has_failure = any(
                            ann.get(f'annotator_{j}', False) 
                            for j in range(1, 10)  # Check annotator_1 through annotator_9
                        )
                        
                        if has_failure:
                            # Extract failure mode from description
                            failure_text = ann.get('failure mode', '')
                            # Parse failure code like "1.1", "2.3", "3.2" etc
                            if failure_text:
                                # Extract code like "1.1 Poor task constraint" -> "SPEC-1"
                                code = self._parse_failure_code(failure_text, i)
                                if code:
                                    codes.append(code)
        
        # Try other possible field names as fallback
        for field in ['mast_failures', 'failure_modes', 'failures']:
            if field in example:
                field_data = example[field]
                if isinstance(field_data, list):
                    for ann in field_data:
                        if isinstance(ann, dict):
                            code = ann.get('code') or ann.get('failure_code') or ann.get('mode')
                            if code:
                                codes.append(code)
                        elif isinstance(ann, str):
                            codes.append(ann)
        
        return codes
    
    def _parse_failure_code(self, failure_text: str, index: int) -> Optional[str]:
        """Parse failure code from MAD dataset failure mode text.
        
        Converts text like "1.1 Poor task constraint compliance" to "SPEC-1"
        """
        import re
        
        # Map MAD numbering to MAST codes
        # Category 1.X = SPEC (Specification), 2.X = SPEC (Step issues), 
        # 3.X = INTER (Inter-agent), 4.X = TASK (Task verification)
        category_map = {
            '1.1': 'SPEC-1',   # Poor task constraint compliance
            '1.2': 'SPEC-2',   # Inconsistency between reasoning and action
            '1.3': 'SPEC-3',   # Undetected conversation ambiguities  
            '1.4': 'SPEC-4',   # Fail to elicit clarification
            '1.5': 'SPEC-5',   # Unaware of stopping conditions
            '2.1': 'SPEC-6',   # Unbatched repetitive execution
            '2.2': 'SPEC-7',   # Step repetition
            '2.3': 'SPEC-8',   # Backtracking interruption
            '2.4': 'SPEC-9',   # Conversation reset
            '2.5': 'INTER-3',  # Derailment from task
            '2.6': 'INTER-4',  # Disobey role specification
            '3.1': 'INTER-1',  # Disagreement induced inaction
            '3.2': 'INTER-2',  # Withholding relevant information
            '3.3': 'INTER-5',  # Ignoring suggestions from agents
            '3.4': 'INTER-6',  # Waiting for known information
            '4.1': 'TASK-1',   # Premature termination
            '4.2': 'TASK-2',   # Lack of result verification
            '4.3': 'TASK-3',   # Lack of critical verification
        }
        
        # Try to extract the code number like "1.1" or "2.3"
        match = re.match(r'^(\d+\.\d+)', failure_text.strip())
        if match:
            code_num = match.group(1)
            return category_map.get(code_num)
        
        return None
    
    def _extract_trace_text(self, example: Dict) -> str:
        """Extract trace text from an example.
        
        The MAD dataset uses 'trace' field containing the full execution log.
        """
        # Direct trace field (MAD format)
        if 'trace' in example and example['trace']:
            trace = example['trace']
            if isinstance(trace, str):
                # Limit length but keep meaningful content
                return trace[:4000]
            elif isinstance(trace, list):
                return "\\n".join(str(t)[:500] for t in trace[:15])
        
        # Try other field names
        for field in ['trace_text', 'execution_trace', 'raw_trace']:
            if field in example and example[field]:
                trace = example[field]
                if isinstance(trace, str):
                    return trace[:4000]
                elif isinstance(trace, list):
                    return "\\n".join(str(t)[:500] for t in trace[:15])
        
        # Fall back to building from metadata
        trace_parts = []
        if 'mas_name' in example:
            trace_parts.append(f"MAS: {example['mas_name']}")
        if 'benchmark_name' in example:
            trace_parts.append(f"Benchmark: {example['benchmark_name']}")
        if 'round' in example:
            trace_parts.append(f"Round: {example['round']}")
        
        return "\\n".join(trace_parts) if trace_parts else "No trace available"
    
    def get_examples_for_mode(self, failure_code: str, max_examples: int = 2) -> List[MASTExample]:
        """
        Get examples that demonstrate a specific failure mode.
        
        Args:
            failure_code: The MAST failure code (e.g., "INTER-3")
            max_examples: Maximum examples to return
            
        Returns:
            List of MASTExample objects
        """
        self.load()
        
        raw_examples = self._examples_by_mode.get(failure_code, [])[:max_examples]
        
        return [self._convert_to_mast_example(ex) for ex in raw_examples]
    
    def get_diverse_examples(self, max_total: int = 5) -> List[MASTExample]:
        """
        Get a diverse set of examples covering different failure modes.
        
        Args:
            max_total: Maximum total examples to return
            
        Returns:
            List of MASTExample objects
        """
        self.load()
        
        examples = []
        seen_modes = set()
        
        # Get one example per mode until we reach max
        for mode, mode_examples in self._examples_by_mode.items():
            if len(examples) >= max_total:
                break
            
            if mode not in seen_modes and mode_examples:
                examples.append(mode_examples[0])
                seen_modes.add(mode)
        
        return [self._convert_to_mast_example(ex) for ex in examples]
    
    def _convert_to_mast_example(self, raw: Dict) -> MASTExample:
        """Convert raw dict to MASTExample."""
        failure_codes = self._extract_failure_codes(raw)
        
        # Get failure descriptions if available
        descriptions = []
        for field in ['mast_failures', 'failure_modes', 'failures', 'annotations']:
            if field in raw and isinstance(raw[field], list):
                for ann in raw[field]:
                    if isinstance(ann, dict):
                        desc = ann.get('description') or ann.get('evidence') or ann.get('reasoning')
                        if desc:
                            descriptions.append(desc)
        
        return MASTExample(
            trace_text=self._extract_trace_text(raw),
            failure_codes=failure_codes,
            failure_descriptions=descriptions,
            mas_type=raw.get('mas_type', 'Unknown'),
            benchmark=raw.get('benchmark', 'Unknown'),
            llm_model=raw.get('llm_model', 'Unknown')
        )
    
    def get_few_shot_prompt_section(self, max_examples: int = 3) -> str:
        """
        Generate a few-shot examples section for prompts.
        
        Returns:
            Formatted string with examples for inclusion in prompts.
        """
        examples = self.get_diverse_examples(max_total=max_examples)
        
        if not examples:
            return ""
        
        lines = ["## REAL EXAMPLES FROM MAD DATASET", ""]
        
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append(f"MAS Type: {ex.mas_type}")
            lines.append(f"Trace (excerpt): {ex.trace_text[:500]}...")
            lines.append(f"Detected Failures: {', '.join(ex.failure_codes)}")
            if ex.failure_descriptions:
                lines.append(f"Evidence: {ex.failure_descriptions[0][:200]}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_available_modes(self) -> List[str]:
        """Get list of failure modes that have examples."""
        self.load()
        return sorted(self._examples_by_mode.keys())
    
    def get_mode_example_counts(self) -> Dict[str, int]:
        """Get count of examples per failure mode."""
        self.load()
        return {mode: len(examples) for mode, examples in self._examples_by_mode.items()}
