"""
沙盒生命周期事件的钩子注册，便于调试/监控。
"""

from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List

Hook = Callable[..., None]


class SandboxHookRegistry:
    """简单的事件-回调映射，用于观测沙盒行为。"""
    def __init__(self):
        self._hooks: DefaultDict[str, List[Hook]] = defaultdict(list)

    def register(self, event: str, hook: Hook) -> None:
        self._hooks[event].append(hook)

    def emit(self, event: str, **payload) -> None:
        for hook in self._hooks.get(event, []):
            hook(**payload)


registry = SandboxHookRegistry()


def on(event: str) -> Callable[[Hook], Hook]:
    def decorator(func: Hook) -> Hook:
        registry.register(event, func)
        return func

    return decorator
