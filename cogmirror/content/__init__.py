"""内容库：Python 学科的认知建模内容.

来自 ECOS 已验证的 Python 内容三件套（见 MIGRATION.md 第1类）：
- bloom_goals: Bloom 六层认知目标库
- misconceptions: Python 常见错误模式库
- threshold_concepts: Python 临界概念（liminal）库
"""

from .python_basics import PythonBasicsBloomLibrary, PythonBasicsTopic
from .misconceptions import (
    MisconceptionEntry,
    PythonBasicsMisconceptionLibrary,
    PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR,
)
from .threshold_concepts import (
    TCStatus,
    ThresholdConceptEntry,
    PythonThresholdConceptLibrary,
)

__all__ = [
    "PythonBasicsBloomLibrary",
    "PythonBasicsTopic",
    "MisconceptionEntry",
    "PythonBasicsMisconceptionLibrary",
    "PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR",
    "TCStatus",
    "ThresholdConceptEntry",
    "PythonThresholdConceptLibrary",
]
