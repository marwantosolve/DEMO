"""
MAST Fine-Tuning Utilities.

Helps export evaluation data into formats suitable for fine-tuning Gemini models.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from mas_eval.core.types import EvaluationResult, TraceData
from mas_eval.mast.classifier import MASTClassifier, ClassifierMode

class FineTuningExporter:
    """
    Exports evaluation results for fine-tuning.
    
    Converts traces and their detected failures into JSONL format
    compatible with Vertex AI / Gemini fine-tuning.
    """
    
    def __init__(self, output_dir: str = "fine_tuning_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_batch(
        self,
        results: List[EvaluationResult],
        filename: str = "tuning_dataset.jsonl"
    ) -> str:
        """
        Export a batch of evaluation results.
        
        Args:
            results: List of EvaluationResult objects
            filename: Output filename
            
        Returns:
            Path to the generated file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                tuning_example = self._convert_to_tuning_example(result)
                if tuning_example:
                    f.write(json.dumps(tuning_example) + "\n")
                    
        return output_path
    
    def _convert_to_tuning_example(self, result: EvaluationResult) -> Optional[Dict]:
        """Convert a single result to a tuning example."""
        # We need the original trace text that was classified
        # Since EvaluationResult stores the outcome, we might need to re-construct the input
        # Or ideally, we'd store the formatted trace in result, but for now we reconstruct
        
        # NOTE: This assumes we can reconstruct the exact text used for classification
        # Ideally, we should modify EvaluationResult to store the input prompt or trace text
        
        # For this utility to work effectively, we need to re-format the spans
        # into the text format expected by the model.
        
        # Using a temporary classifier instance just to access the formatting logic
        classifier = MASTClassifier()
        # We need the spans, but EvaluationResult abstracts them.
        # This is a limitation. We need to pass the TraceData or Spans source.
        
        # For now, we will create a generic structure, assuming the user provides
        # the input text if possible, or we skip this automatic reconstruction
        # if the data isn't available.
        
        # As a robust fallback, let's define the structure for the "text" input
        
        # Input: The Trace text
        # Output: The JSON structure of failures
        
        # We'll define the ideal input/output pair
        
        trace_text = "..." # Placeholder: In a real scenario, we'd need the source spans
        
        # Construct the target JSON
        target_json = {
            "failures": [
                {
                    "code": f.code,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                    # We can exclude reasoning to save output tokens if desired,
                    # but including it helps the model learn to reason.
                    "reasoning": f.reasoning 
                }
                for f in result.failures
            ],
            "summary": "Auto-generated training example"
        }
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Analyze the MAS trace for MAST failure modes. Respond with JSON."
                },
                {
                    "role": "user",
                    "content": trace_text 
                },
                {
                    "role": "model",
                    "content": json.dumps(target_json)
                }
            ]
        }

    @staticmethod
    def prepare_from_mad_dataset(
        mad_dataset_path: str,
        output_path: str,
        limit: int = 100
    ) -> str:
        """
        Convert entries from the MAD dataset directly to tuning format.
        
        This is the preferred way to bootstrap a fine-tuned model.
        """
        try:
            with open(mad_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                data = [data]
                
            # Use MAST classifier to help formatting
            classifier = MASTClassifier()
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                count = 0
                for entry in data:
                    if count >= limit:
                        break
                        
                    # Extract trace
                    trace_text = entry.get('trace', '')
                    if isinstance(trace_text, list):
                        trace_text = "\n".join(str(t) for t in trace_text)
                    
                    if not trace_text:
                        continue
                        
                    # Extract failures (target)
                    failures = []
                    # Logic to extract failures from MAD annotations (reusing logic from mast_fewshot)
                    # For simplicity, we'll assume we can get a list of failure codes/descriptions
                    
                    # ... extraction logic similar to MASTFewShotLoader ...
                    # For this demo, we'll create a simplified target
                    
                    target = {
                        "analysis": {
                            "agents_identified": [],
                            "task_objective": "Unknown",
                            "key_observations": []
                        },
                        "failures": [], # Populate this properly
                        "summary": "Training example"
                    }
                    
                    example = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": classifier._fine_tuned_prompt("").split("\nTrace:")[0] # Get system prompt
                            },
                            {
                                "role": "user",
                                "content": trace_text
                            },
                            {
                                "role": "model",
                                "content": json.dumps(target)
                            }
                        ]
                    }
                    
                    f_out.write(json.dumps(example) + "\n")
                    count += 1
                    
            return output_path
            
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return ""

