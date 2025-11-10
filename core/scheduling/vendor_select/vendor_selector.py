"""
Backward-compatible shim that exposes the new VendorSelector implementation.
"""

from core.scheduling.vendor_selector import VendorDecision, VendorSelector

__all__ = ["VendorSelector", "VendorDecision"]
