"""Reading GitHub into artifacts.

Separate from `connectors/enterprise/github.py`, which reads issues, discussions
and releases into `Signal`s. See `reader.py` for why the two paths are distinct
rather than one connector with two outputs.
"""
