"""
跨厂商/资源池选址模块

实现一级调度：基于折算系数与平台得分选择最佳GPU平台
"""

from .vendor_selector import VendorSelector, Platform, VendorSelectionResult

__all__ = ['VendorSelector', 'Platform', 'VendorSelectionResult']

