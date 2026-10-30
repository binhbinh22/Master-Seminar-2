"""Progress bars in terminals; clean epoch logs when redirected to a file."""

import sys
import os

from tqdm.auto import tqdm


def progress(iterable, *, enabled=True, **kwargs):
    mode = os.environ.get("MODEL_ZOO_PROGRESS", "auto")
    visible = mode == "on" or (mode != "off" and sys.stderr.isatty())
    return tqdm(iterable, disable=not enabled or not visible,
                dynamic_ncols=True, leave=False, mininterval=1.0, **kwargs)
