__all__ = ["__version__"]

try:
	from ._version import version as __version__
except Exception:
	# During development or when package metadata is not generated, fall back to a default.
	__version__ = "0+unknown"
