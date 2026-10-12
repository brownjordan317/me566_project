"""Progress output shared by every stage.

On a terminal, loops get tqdm bars and long opaque steps get halo spinners.
When stderr is a log file (benchmark runs) or NO_PROGRESS=1 is set, both fall
back to one plain line when a step starts and one when it ends, so logs stay
readable. Everything goes to stderr; stdout stays free for results.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Iterable, Iterator

from tqdm import tqdm


def interactive() -> bool:
    return sys.stderr.isatty() and not os.environ.get('NO_PROGRESS')


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _LogBar:
    """tqdm stand-in for logs: a start line, then a summary when closed."""

    def __init__(self, iterable, desc: str, total, unit: str):
        self.iterable, self.desc, self.unit = iterable, desc, unit
        self.total = total if total is not None else (
            len(iterable) if hasattr(iterable, '__len__') else None)
        self.n, self.postfix, self.start = 0, {}, time.monotonic()
        _log(f'{desc}: started' + (f' ({self.total} {unit})' if self.total else ''))

    def __iter__(self) -> Iterator:
        try:
            for item in self.iterable:
                yield item
                self.n += 1
        finally:
            self.close()

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, **values) -> None:
        self.postfix = values

    def close(self) -> None:
        if self.start is None:
            return
        extra = ''.join(f', {key} {value}' for key, value in self.postfix.items())
        _log(f'{self.desc}: done, {self.n} {self.unit} in {time.monotonic() - self.start:.1f} s{extra}')
        self.start = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def bar(iterable: Iterable | None = None, desc: str = '', total: int | None = None,
        unit: str = 'it'):
    """A tqdm progress bar, or a two-line log summary when not on a terminal."""
    if interactive():
        return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True,
                    file=sys.stderr)
    return _LogBar(iterable if iterable is not None else (), desc, total, unit)


@contextlib.contextmanager
def spinner(text: str):
    """A halo spinner around a step without measurable progress (COLMAP, model loading)."""
    start = time.monotonic()
    if interactive():
        from halo import Halo
        spin = Halo(text=text, spinner='dots', stream=sys.stderr)
        spin.start()
        try:
            yield spin
        except BaseException:
            spin.fail(f'{text} (failed after {time.monotonic() - start:.0f} s)')
            raise
        spin.succeed(f'{text} ({time.monotonic() - start:.1f} s)')
        return
    _log(f'{text}...')
    try:
        yield None
    except BaseException:
        _log(f'{text}: failed after {time.monotonic() - start:.1f} s')
        raise
    _log(f'{text}: done in {time.monotonic() - start:.1f} s')


def done(message: str) -> None:
    """One summary line when a stage finishes; full details are in its JSON report."""
    from rich.console import Console
    Console(stderr=True, highlight=False).print(f'[green]✓[/green] {message}')


def heading(text: str, detail: str = '') -> None:
    """A stage label matching the Makefile's '▶ <stage>' lines."""
    print(f'\033[1;36m▶ {text}\033[0m' + (f'  {detail}' if detail else ''),
          file=sys.stderr, flush=True)
