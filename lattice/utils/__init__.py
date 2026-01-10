"""Utilities package for Lattice."""

__all__: list[str] = []


def _lazy_import(name: str):
    import importlib

    module_path, attr = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def __getattr__(name: str):
    if name == "PlaceholderInjector":
        return _lazy_import("lattice.utils.placeholder_injector.PlaceholderInjector")
    if name == "PlaceholderDef":
        return _lazy_import("lattice.utils.placeholder_registry.PlaceholderDef")
    if name == "PlaceholderRegistry":
        return _lazy_import("lattice.utils.placeholder_registry.PlaceholderRegistry")
    if name == "get_registry":
        return _lazy_import("lattice.utils.placeholder_registry.get_registry")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
