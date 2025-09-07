"""
Asymmetric HRM Model - System 1/System 2 Architecture

This module implements a Kahneman-inspired hierarchical reasoning model where:
- System 1 (Low-level): Fast, intuitive processing like an LLM
- System 2 (High-level): Deliberate, analytical processing with memory

Based on the original HRM architecture but with specialized modules for different
types of reasoning.
"""

from .asymmetric_hrm import AsymmetricHRMModel, create_asymmetric_hrm_model
from .system1_module import System1Module
from .system2_module import System2Module

__all__ = [
    "AsymmetricHRMModel",
    "create_asymmetric_hrm_model",
    "System1Module",
    "System2Module",
]
