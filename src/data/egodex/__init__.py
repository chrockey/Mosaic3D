"""EgoDex corpus. `egodex_io` is framework-free on purpose -- CPU probes import it without torch.

EgoDexClipDataset is exposed lazily so that importing this package in a torch-free venv works.
"""


def __getattr__(name):
    if name == "EgoDexClipDataset":
        from src.data.egodex.egodex_clip_dataset import EgoDexClipDataset

        return EgoDexClipDataset
    raise AttributeError(name)


__all__ = ["EgoDexClipDataset"]
