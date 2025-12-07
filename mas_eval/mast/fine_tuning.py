"""
MAST Fine-Tuning Module.

Provides REAL fine-tuning of Gemini models on the MAST dataset using
Google's tunedModels.create API. This actually trains the model weights,
not just in-context learning.

Usage:
    from mas_eval.mast.fine_tuning import MASTFineTuner
    
    # Train once (takes 5-30 min)
    tuner = MASTFineTuner(api_key="your-key")
    tuned_model = tuner.train_on_mad_dataset("mast_dataset/MAD_human_labelled_dataset.json")
    
    # Use the tuned model
    classifier = MASTClassifier(model=tuned_model, mode=ClassifierMode.FINE_TUNED)
"""

import json
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    base_model: str = "models/gemini-1.5-flash-001-tuning"  # Tunable base model
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 0.001
    display_name: str = "mast-classifier"


@dataclass
class TrainingExample:
    """A single training example."""
    text_input: str
    output: str


class MASTFineTuner:
    """
    Fine-tunes Gemini models on MAST failure detection.
    
    This class handles:
    1. Preparing training data from MAD dataset
    2. Creating fine-tuning jobs via Gemini API
    3. Monitoring training progress
    4. Returning the tuned model ID for use
    
    Example:
        tuner = MASTFineTuner(api_key="your-key")
        model_id = tuner.train_on_mad_dataset("path/to/MAD_human_labelled_dataset.json")
        # model_id is like "tunedModels/mast-classifier-abc123"
    """
    
    # Mapping from MAD dataset codes to MAST codes
    MAD_TO_MAST_CODES = {
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
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            api_key: Gemini API key (or uses GOOGLE_API_KEY env var)
            config: Training configuration
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.config = config or TrainingConfig()
        self._client = None
        self._training_examples: List[TrainingExample] = []
        self._tuned_model_name: Optional[str] = None
        
    def _initialize_client(self):
        """Initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                
                if not self.api_key:
                    raise ValueError(
                        "Google API key required. Set GOOGLE_API_KEY environment variable "
                        "or pass api_key to constructor."
                    )
                
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required. Install with: pip install google-generativeai"
                )
    
    def prepare_training_data(
        self, 
        mad_dataset_path: str,
        max_examples: int = 500,
        min_examples: int = 10
    ) -> int:
        """
        Prepare training data from MAD dataset.
        
        Args:
            mad_dataset_path: Path to MAD JSON file
            max_examples: Maximum training examples
            min_examples: Minimum required examples
            
        Returns:
            Number of examples prepared
        """
        print(f"📚 Loading MAD dataset from {mad_dataset_path}...")
        
        with open(mad_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        print(f"   Found {len(data)} entries in dataset")
        
        self._training_examples = []
        
        for entry in data:
            if len(self._training_examples) >= max_examples:
                break
            
            example = self._convert_mad_entry(entry)
            if example:
                self._training_examples.append(example)
        
        count = len(self._training_examples)
        print(f"✅ Prepared {count} training examples")
        
        if count < min_examples:
            print(f"⚠️  Warning: Only {count} examples. Need at least {min_examples} for good results.")
        
        return count
    
    def _convert_mad_entry(self, entry: Dict) -> Optional[TrainingExample]:
        """Convert a MAD dataset entry to training example."""
        # Extract trace text
        trace = entry.get('trace', '')
        if isinstance(trace, list):
            trace = "\n".join(str(t) for t in trace)
        
        if not trace or len(trace) < 50:
            return None
        
        # Truncate very long traces
        if len(trace) > 4000:
            trace = trace[:4000] + "\n[... trace truncated ...]"
        
        # Extract failure annotations
        failures = []
        
        if 'annotations' in entry:
            annotations = entry['annotations']
            if isinstance(annotations, list):
                for ann in annotations:
                    if isinstance(ann, dict):
                        # Check if any annotator marked this as failure
                        has_failure = any(
                            ann.get(f'annotator_{j}', False) 
                            for j in range(1, 10)
                        )
                        
                        if has_failure:
                            failure_text = ann.get('failure mode', '')
                            if failure_text:
                                # Parse failure code
                                match = re.match(r'^(\d+\.\d+)', failure_text.strip())
                                if match:
                                    code_num = match.group(1)
                                    mast_code = self.MAD_TO_MAST_CODES.get(code_num)
                                    if mast_code:
                                        # Extract description
                                        desc = failure_text[len(code_num):].strip()
                                        failures.append({
                                            "code": mast_code,
                                            "confidence_breakdown": {
                                                "evidence_strength": 0.9,
                                                "pattern_match_score": 0.85,
                                                "cross_reference_score": 0.8,
                                                "model_confidence": 0.9
                                            },
                                            "evidence": desc or f"Annotated failure: {mast_code}",
                                            "proof_quotes": [trace[:200]],
                                            "affected_steps": [],
                                            "reasoning_chain": f"Human annotator marked this as {mast_code}"
                                        })
        
        # Create target output
        target = {
            "analysis": {
                "agents_identified": [],
                "task_objective": "Extracted from trace",
                "key_observations": []
            },
            "failures": failures,
            "summary": f"Found {len(failures)} failure(s)" if failures else "No failures detected"
        }
        
        # Format input text
        input_text = f"Analyze the following MAS trace for MAST failure modes:\n\n{trace}"
        
        return TrainingExample(
            text_input=input_text,
            output=json.dumps(target)
        )
    
    def start_training(self, wait: bool = True) -> str:
        """
        Start the fine-tuning job.
        
        Args:
            wait: If True, wait for training to complete
            
        Returns:
            The tuned model name (e.g., "tunedModels/mast-classifier-xxx")
        """
        self._initialize_client()
        
        if not self._training_examples:
            raise ValueError("No training data. Call prepare_training_data() first.")
        
        print(f"🚀 Starting fine-tuning with {len(self._training_examples)} examples...")
        print(f"   Base model: {self.config.base_model}")
        print(f"   Epochs: {self.config.epochs}")
        
        # Convert to Gemini format
        training_data = [
            {"text_input": ex.text_input, "output": ex.output}
            for ex in self._training_examples
        ]
        
        try:
            # Create tuning job
            operation = self._client.create_tuned_model(
                display_name=self.config.display_name,
                source_model=self.config.base_model,
                epoch_count=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                training_data=training_data
            )
            
            print(f"✅ Tuning job created!")
            print(f"   Job name: {operation.metadata.tuned_model}")
            
            if wait:
                return self.wait_for_training(operation)
            else:
                self._tuned_model_name = operation.metadata.tuned_model
                print(f"   Training in progress. Use wait_for_training() to monitor.")
                return self._tuned_model_name
                
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            raise
    
    def wait_for_training(self, operation=None) -> str:
        """
        Wait for training to complete.
        
        Args:
            operation: The training operation (optional if already started)
            
        Returns:
            The tuned model name
        """
        if operation is None:
            print("⚠️  No operation provided. Cannot wait.")
            return self._tuned_model_name or ""
        
        print("⏳ Waiting for training to complete...")
        print("   (This may take 5-30 minutes)")
        
        start_time = time.time()
        
        for status in operation.wait_bar():
            elapsed = time.time() - start_time
            print(f"   Status: {status.state.name} ({elapsed:.0f}s elapsed)")
        
        # Get the result
        result = operation.result()
        self._tuned_model_name = result.name
        
        elapsed = time.time() - start_time
        print(f"✅ Training complete! ({elapsed:.1f}s)")
        print(f"   Tuned model: {self._tuned_model_name}")
        print(f"\n💡 Save this model name for future use:")
        print(f'   model="{self._tuned_model_name}"')
        
        return self._tuned_model_name
    
    def train_on_mad_dataset(
        self, 
        mad_dataset_path: str,
        max_examples: int = 200
    ) -> str:
        """
        One-step method: prepare data and train.
        
        Args:
            mad_dataset_path: Path to MAD dataset JSON
            max_examples: Max training examples
            
        Returns:
            The tuned model name
        """
        self.prepare_training_data(mad_dataset_path, max_examples=max_examples)
        return self.start_training(wait=True)
    
    def get_tuned_model_name(self) -> Optional[str]:
        """Get the name of the tuned model (after training)."""
        return self._tuned_model_name
    
    def list_tuned_models(self) -> List[Dict[str, Any]]:
        """List all tuned models in the account."""
        self._initialize_client()
        
        models = []
        for model in self._client.list_tuned_models():
            models.append({
                "name": model.name,
                "display_name": model.display_name,
                "state": model.state.name,
                "create_time": str(model.create_time),
                "base_model": model.base_model
            })
        
        return models
    
    def export_training_data(self, output_path: str) -> str:
        """
        Export prepared training data to JSONL file.
        
        Useful for manual upload or debugging.
        """
        if not self._training_examples:
            print("⚠️  No training data. Call prepare_training_data() first.")
            return ""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in self._training_examples:
                f.write(json.dumps({
                    "text_input": ex.text_input,
                    "output": ex.output
                }) + "\n")
        
        print(f"✅ Exported {len(self._training_examples)} examples to {output_path}")
        return output_path


# Legacy class for backward compatibility
class FineTuningExporter:
    """Legacy exporter - use MASTFineTuner instead."""
    
    def __init__(self, output_dir: str = "fine_tuning_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("⚠️  FineTuningExporter is deprecated. Use MASTFineTuner for real fine-tuning.")
