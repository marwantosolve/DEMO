"""
MAST Fine-Tuning Module.

Provides REAL fine-tuning of Gemini models on the MAST dataset using
Google's tunedModels.create API. This actually trains the model weights,
not just in-context learning.

Enhanced Features:
- Training progress tracking with metrics
- Training results visualization  
- Model config save/load for persistence
- Easy integration with MASTClassifier

Usage:
    from mas_eval.mast.fine_tuning import MASTFineTuner
    
    # Train once (takes 5-30 min)
    tuner = MASTFineTuner(api_key="your-key")
    result = tuner.train_on_mad_dataset("mast_dataset/MAD_human_labelled_dataset.json")
    
    # Display results and save
    tuner.display_training_results(result)
    tuner.save_model_config("tuned_model.json")
    
    # Later: load and use
    from mas_eval.mast import MASTClassifier, ClassifierMode
    classifier = MASTClassifier(model=result.model_name, mode=ClassifierMode.FINE_TUNED)
"""

import json
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Note: Tunable models have specific naming - check API docs for latest
    base_model: str = "models/gemini-2.0-flash-tuning-exp"  # Tunable base model
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 0.001
    display_name: str = "mast-classifier"


@dataclass
class TrainingExample:
    """A single training example."""
    text_input: str
    output: str


@dataclass
class TrainingProgress:
    """Tracks training progress during fine-tuning."""
    epoch: int
    total_epochs: int
    elapsed_seconds: float
    status: str
    loss: Optional[float] = None
    
    def __str__(self) -> str:
        loss_str = f", loss={self.loss:.4f}" if self.loss else ""
        return f"Epoch {self.epoch}/{self.total_epochs} ({self.status}{loss_str}) - {self.elapsed_seconds:.0f}s"


@dataclass
class TrainingResult:
    """Complete result of a fine-tuning job."""
    model_name: str
    display_name: str
    base_model: str
    training_examples: int
    epochs: int
    total_time_seconds: float
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    final_loss: Optional[float] = None
    created_at: str = ""
    dataset_path: str = ""
    status: str = "completed"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "base_model": self.base_model,
            "training_examples": self.training_examples,
            "epochs": self.epochs,
            "total_time_seconds": self.total_time_seconds,
            "final_loss": self.final_loss,
            "snapshots": self.snapshots,
            "created_at": self.created_at,
            "dataset_path": self.dataset_path,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingResult":
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", ""),
            display_name=data.get("display_name", ""),
            base_model=data.get("base_model", ""),
            training_examples=data.get("training_examples", 0),
            epochs=data.get("epochs", 0),
            total_time_seconds=data.get("total_time_seconds", 0),
            final_loss=data.get("final_loss"),
            snapshots=data.get("snapshots", []),
            created_at=data.get("created_at", ""),
            dataset_path=data.get("dataset_path", ""),
            status=data.get("status", "unknown")
        )


class MASTFineTuner:
    """
    Fine-tunes Gemini models on MAST failure detection.
    
    This class handles:
    1. Preparing training data from MAD dataset
    2. Creating fine-tuning jobs via Gemini API
    3. Monitoring training progress with metrics
    4. Saving and loading model configurations
    5. Displaying training results
    
    Example:
        tuner = MASTFineTuner(api_key="your-key")
        result = tuner.train_on_mad_dataset("path/to/MAD_human_labelled_dataset.json")
        
        # Display results
        tuner.display_training_results(result)
        
        # Save for later use
        tuner.save_model_config("tuned_model.json")
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
        self._last_result: Optional[TrainingResult] = None
        self._dataset_path: str = ""
        
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
        self._dataset_path = mad_dataset_path
        
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
    
    def start_training(self, wait: bool = True) -> TrainingResult:
        """
        Start the fine-tuning job.
        
        Args:
            wait: If True, wait for training to complete
            
        Returns:
            TrainingResult with full training information
        """
        self._initialize_client()
        
        if not self._training_examples:
            raise ValueError("No training data. Call prepare_training_data() first.")
        
        print(f"🚀 Starting fine-tuning with {len(self._training_examples)} examples...")
        print(f"   Base model: {self.config.base_model}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        
        # Convert to Gemini format
        training_data = [
            {"text_input": ex.text_input, "output": ex.output}
            for ex in self._training_examples
        ]
        
        start_time = time.time()
        snapshots = []
        
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
                result = self._wait_with_progress(operation, start_time, snapshots)
                self._last_result = result
                return result
            else:
                self._tuned_model_name = operation.metadata.tuned_model
                result = TrainingResult(
                    model_name=self._tuned_model_name,
                    display_name=self.config.display_name,
                    base_model=self.config.base_model,
                    training_examples=len(self._training_examples),
                    epochs=self.config.epochs,
                    total_time_seconds=time.time() - start_time,
                    dataset_path=self._dataset_path,
                    status="in_progress"
                )
                self._last_result = result
                print(f"   Training in progress. Use wait_for_training() to monitor.")
                return result
                
        except Exception as e:
            print(f"❌ Error starting training: {e}")
            raise
    
    def _wait_with_progress(
        self, 
        operation, 
        start_time: float,
        snapshots: List[Dict]
    ) -> TrainingResult:
        """Wait for training with progress tracking."""
        print("\n⏳ Training in progress...")
        print("   (This may take 5-30 minutes)\n")
        
        last_epoch = 0
        
        try:
            for status in operation.wait_bar():
                elapsed = time.time() - start_time
                
                # Extract training progress
                progress_info = {
                    "elapsed_seconds": elapsed,
                    "status": status.state.name if hasattr(status, 'state') else str(status)
                }
                
                # Try to get training metrics
                if hasattr(status, 'metadata'):
                    meta = status.metadata
                    if hasattr(meta, 'tuning_task'):
                        task = meta.tuning_task
                        if hasattr(task, 'snapshots') and task.snapshots:
                            for snap in task.snapshots:
                                epoch = getattr(snap, 'epoch', len(snapshots) + 1)
                                if epoch > last_epoch:
                                    last_epoch = epoch
                                    snap_data = {
                                        "epoch": epoch,
                                        "elapsed_seconds": elapsed
                                    }
                                    if hasattr(snap, 'mean_loss'):
                                        snap_data["loss"] = snap.mean_loss
                                        print(f"   📊 Epoch {epoch}: loss={snap.mean_loss:.4f} ({elapsed:.0f}s)")
                                    else:
                                        print(f"   📊 Epoch {epoch} complete ({elapsed:.0f}s)")
                                    snapshots.append(snap_data)
                
                # Print status update periodically
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    state_name = status.state.name if hasattr(status, 'state') else 'RUNNING'
                    print(f"   ⏳ Status: {state_name} ({elapsed:.0f}s elapsed)")
        
        except Exception as e:
            print(f"   ⚠️ Progress tracking error: {e}")
        
        # Get the final result
        try:
            result_model = operation.result()
            self._tuned_model_name = result_model.name
        except Exception as e:
            # Try to get model name from operation
            if hasattr(operation, 'metadata') and hasattr(operation.metadata, 'tuned_model'):
                self._tuned_model_name = operation.metadata.tuned_model
            else:
                raise e
        
        total_time = time.time() - start_time
        
        # Calculate final loss from snapshots
        final_loss = None
        if snapshots:
            last_snap = snapshots[-1]
            final_loss = last_snap.get("loss")
        
        result = TrainingResult(
            model_name=self._tuned_model_name,
            display_name=self.config.display_name,
            base_model=self.config.base_model,
            training_examples=len(self._training_examples),
            epochs=self.config.epochs,
            total_time_seconds=total_time,
            final_loss=final_loss,
            snapshots=snapshots,
            dataset_path=self._dataset_path,
            status="completed"
        )
        
        print(f"\n✅ Training complete! ({total_time:.1f}s)")
        print(f"   Model: {self._tuned_model_name}")
        
        return result
    
    def wait_for_training(self, operation=None) -> TrainingResult:
        """
        Wait for training to complete.
        
        Args:
            operation: The training operation (optional if already started)
            
        Returns:
            TrainingResult with full information
        """
        if operation is None:
            print("⚠️  No operation provided. Cannot wait.")
            return self._last_result or TrainingResult(
                model_name=self._tuned_model_name or "",
                display_name=self.config.display_name,
                base_model=self.config.base_model,
                training_examples=len(self._training_examples),
                epochs=self.config.epochs,
                total_time_seconds=0,
                status="unknown"
            )
        
        return self._wait_with_progress(operation, time.time(), [])
    
    def train_on_mad_dataset(
        self, 
        mad_dataset_path: str,
        max_examples: int = 200
    ) -> TrainingResult:
        """
        One-step method: prepare data and train.
        
        Args:
            mad_dataset_path: Path to MAD dataset JSON
            max_examples: Max training examples
            
        Returns:
            TrainingResult with full training information
        """
        self.prepare_training_data(mad_dataset_path, max_examples=max_examples)
        return self.start_training(wait=True)
    
    def display_training_results(self, result: TrainingResult = None) -> None:
        """
        Display training results with formatting.
        
        Args:
            result: TrainingResult to display (uses last result if None)
        """
        result = result or self._last_result
        if not result:
            print("❌ No training results to display.")
            return
        
        # Format time
        mins = int(result.total_time_seconds // 60)
        secs = int(result.total_time_seconds % 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print("\n" + "╔" + "═" * 62 + "╗")
        print("║" + "🎯 MAST FINE-TUNING COMPLETE".center(62) + "║")
        print("╠" + "═" * 62 + "╣")
        print(f"║ Model Name: {result.model_name[:48]:<48} ║")
        print(f"║ Display Name: {result.display_name:<46} ║")
        print(f"║ Base Model: {result.base_model:<48} ║")
        print(f"║ Training Examples: {result.training_examples:<41} ║")
        print(f"║ Epochs: {result.epochs:<52} ║")
        print(f"║ Total Time: {time_str:<48} ║")
        
        if result.final_loss is not None:
            print(f"║ Final Loss: {result.final_loss:<48.4f} ║")
        
        if result.snapshots:
            print("╠" + "═" * 62 + "╣")
            print("║ 📊 Training Progress:".ljust(63) + "║")
            for snap in result.snapshots:
                epoch = snap.get("epoch", "?")
                loss = snap.get("loss")
                loss_str = f"loss={loss:.4f}" if loss else "complete"
                print(f"║   Epoch {epoch}: {loss_str}".ljust(63) + "║")
        
        print("╠" + "═" * 62 + "╣")
        print("║ 💡 Usage:".ljust(63) + "║")
        print(f'║   classifier = MASTClassifier('.ljust(63) + "║")
        print(f'║       model="{result.model_name[:40]}...",'.ljust(63) + "║")
        print(f'║       mode=ClassifierMode.FINE_TUNED'.ljust(63) + "║")
        print(f'║   )'.ljust(63) + "║")
        print("╚" + "═" * 62 + "╝")
    
    def save_model_config(self, output_path: str = "tuned_model.json") -> str:
        """
        Save tuned model configuration to JSON file.
        
        Args:
            output_path: Path to save the config file
            
        Returns:
            Path to saved config file
        """
        result = self._last_result
        if not result:
            print("❌ No training result to save. Run training first.")
            return ""
        
        config = result.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Model config saved to: {output_path}")
        print(f"   Model: {result.model_name}")
        return output_path
    
    @classmethod
    def load_model_config(cls, config_path: str) -> TrainingResult:
        """
        Load tuned model configuration from JSON file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            TrainingResult with loaded configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = TrainingResult.from_dict(data)
        print(f"✅ Loaded model config from: {config_path}")
        print(f"   Model: {result.model_name}")
        return result
    
    def get_tuned_model_name(self) -> Optional[str]:
        """Get the name of the tuned model (after training)."""
        return self._tuned_model_name
    
    def get_last_result(self) -> Optional[TrainingResult]:
        """Get the last training result."""
        return self._last_result
    
    def list_tuned_models(self) -> List[Dict[str, Any]]:
        """List all tuned models in the account."""
        self._initialize_client()
        
        models = []
        try:
            for model in self._client.list_tuned_models():
                models.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "state": model.state.name if hasattr(model.state, 'name') else str(model.state),
                    "create_time": str(model.create_time) if hasattr(model, 'create_time') else "",
                    "base_model": model.base_model if hasattr(model, 'base_model') else ""
                })
        except Exception as e:
            print(f"⚠️ Error listing models: {e}")
        
        return models
    
    def display_tuned_models(self) -> None:
        """Display all tuned models in a formatted table."""
        models = self.list_tuned_models()
        
        if not models:
            print("📭 No tuned models found.")
            return
        
        print("\n" + "=" * 80)
        print("📋 AVAILABLE TUNED MODELS")
        print("=" * 80)
        
        for i, model in enumerate(models, 1):
            print(f"\n[{i}] {model.get('display_name', 'Unknown')}")
            print(f"    Name: {model.get('name', 'N/A')}")
            print(f"    Status: {model.get('state', 'N/A')}")
            print(f"    Base: {model.get('base_model', 'N/A')}")
            print(f"    Created: {model.get('create_time', 'N/A')}")
        
        print("\n" + "=" * 80)
    
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
