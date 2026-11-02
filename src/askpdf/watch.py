"""Folder watching (ADR-0009): poll the folder and index what has settled.

Polling with ``os.scandir`` behaves the same on every OS and on iCloud or network
folders, needs no dependency, and is cheap at the scale askpdf is for (hundreds of
PDFs). A file is read only once its size and modification time are the same in two
scans in a row, so a file still being copied is never indexed half-written.
"""

import threading
from collections.abc import Callable
from pathlib import Path

from askpdf.indexer import scan

POLL_SECONDS = 2.0

# relative path -> (size, mtime_ns); (-1, -1) for a file that cannot be stat'ed
Listing = dict[str, tuple[int, int]]


def listing(root: Path, exclude: frozenset[Path] = frozenset()) -> Listing:
    """Size and modification time of every PDF the indexer would see."""
    found: Listing = {}
    for rel, real in scan(root, None, exclude):
        try:
            info = real.stat()
        except FileNotFoundError:
            continue  # deleted since it was listed
        except OSError:
            found[rel] = (-1, -1)  # unreadable: the indexer reports it
            continue
        found[rel] = (info.st_size, info.st_mtime_ns)
    return found


class Watcher:
    """Decides, scan by scan, when to index and which files to leave for later.

    It holds no clock and touches no files: ``tick`` takes the latest listing, so tests
    drive it scan by scan.
    """

    def __init__(self, first: Listing) -> None:
        self.last = first
        self.pending = False

    def tick(self, now: Listing) -> frozenset[str] | None:
        """None if there is nothing to do; otherwise index now, skipping these paths.

        The paths returned changed since the previous scan, so they may still be being
        written. Everything else that changed has settled (or is gone) and is indexed
        now; the skipped ones are picked up on a later scan, once they hold still.
        """
        unsettled = frozenset(path for path, seen in now.items() if self.last.get(path) != seen)
        if now != self.last:
            self.pending = True
        self.last = now
        if not self.pending:
            return None
        self.pending = bool(unsettled)
        return unsettled

    def retry(self) -> None:
        """Indexing failed (the index was busy, say): try again on the next scan."""
        self.pending = True


def watch(
    root: Path,
    exclude: frozenset[Path],
    index: Callable[[frozenset[str]], bool],
    stop: threading.Event,
    first: Listing,
    interval: float = POLL_SECONDS,
) -> None:
    """Poll ``root`` until ``stop`` is set, calling ``index(skip)`` when something settled.

    ``first`` is the listing taken before the start-up indexing run, so a change made
    while that ran is still seen. ``index`` reports its own errors and returns False if
    it did not finish, to be tried again on the next scan.
    """
    watcher = Watcher(first)
    while not stop.wait(interval):
        skip = watcher.tick(listing(root, exclude))
        if skip is not None and not index(skip):
            watcher.retry()
