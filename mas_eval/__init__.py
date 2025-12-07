# MAS Evaluation Framework
# A modular, plug-and-play framework for evaluating Multi-Agent Systems

from mas_eval.evaluator import MASEvaluator
from mas_eval.tracer import TracerModule
from mas_eval.graph import CRGModule
from mas_eval.metrics import MetricsModule, GEMMAS_Evaluator, ThoughtRelevanceMetric
from mas_eval.mast import MASTClassifier, MASTTaxonomy
from mas_eval.suggestions import MASAdvisor

__version__ = "0.2.0"
__all__ = [
    "MASEvaluator",
    "TracerModule", 
    "CRGModule",
    "MetricsModule",
    "GEMMAS_Evaluator",
    "MASTClassifier",
    "MASTTaxonomy",
    "ThoughtRelevanceMetric",
    "MASAdvisor"
]
