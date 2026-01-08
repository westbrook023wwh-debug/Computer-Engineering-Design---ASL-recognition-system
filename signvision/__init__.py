from .config import SignVisionConfig

__all__ = ["SignVisionConfig", "SignVisionModel"]


def __getattr__(name: str):
    if name == "SignVisionModel":
        from .model import SignVisionModel

        return SignVisionModel
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
