"""
Compat layer that re-exports the concrete NodeSelector implementation.
"""

from core.scheduling.node_selector import NodeScore, NodeSelector

__all__ = ["NodeSelector", "NodeScore"]
