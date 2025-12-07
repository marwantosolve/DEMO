
import sys
import os

# Add current dir to path to find mas_eval
sys.path.append(os.getcwd())

from mas_eval.evaluator import MASEvaluator, ClassifierMode
from mas_eval.mast.fine_tuning import FineTuningExporter

def test_imports_and_config():
    print("Testing imports and configuration...")
    
    # Test 1: Initialize with default config
    evaluator = MASEvaluator(
        enable_mast=True,
        mast_mode="few_shot_icl"
    )
    
    mast = evaluator.get_mast()
    assert mast is not None
    assert mast.mode == ClassifierMode.FEW_SHOT_ICL, f"Expected FEW_SHOT_ICL, got {mast.mode}"
    print("✅ Default initialization passed (FEW_SHOT_ICL)")
    
    # Test 2: Initialize with Fine-Tuned mode
    evaluator_ft = MASEvaluator(
        enable_mast=True,
        mast_mode="fine_tuned",
        mast_model="projects/my-project/locations/us-central1/endpoints/12345"
    )
    
    mast_ft = evaluator_ft.get_mast()
    assert mast_ft.mode == ClassifierMode.FINE_TUNED, f"Expected FINE_TUNED, got {mast_ft.mode}"
    print("✅ Fine-tuned initialization passed")
    
    # Test 3: Fine-tuning exporter
    exporter = FineTuningExporter()
    assert os.path.exists(exporter.output_dir)
    print("✅ Fine-tuning exporter initialized")

    print("\nAll verification tests passed!")

if __name__ == "__main__":
    test_imports_and_config()
