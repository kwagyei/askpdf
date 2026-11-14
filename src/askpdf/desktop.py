"""The macOS app's side of askpdf: the folder dialog and its alerts (ADR-0013).

The dialog runs in its own ``osascript`` process, so uvicorn keeps the main thread (Tk
would need it). Text reaches the script as arguments, never spliced into its source, so
a folder or message can't change what the script does.
"""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

PROMPT = "Choose the folder of PDFs to ask questions about"
LOG = Path("~/Library/Logs/askpdf/askpdf.log").expanduser()

_CHOOSE = """
on run argv
    activate
    set start to POSIX file (item 2 of argv)
    set chosen to choose folder with prompt (item 1 of argv) default location start
    return POSIX path of chosen
end run
"""
_ALERT = """
on run argv
    activate
    display alert "askpdf" message (item 1 of argv) as warning
end run
"""
_CANCELLED = "-128"  # AppleScript's "User canceled."

type Run = Callable[[list[str]], subprocess.CompletedProcess[str]]


class DialogError(Exception):
    """The folder dialog could not be shown."""


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    # A person is choosing, so there is no timeout. The script is fixed text, and what
    # varies (a prompt, a path) goes in as arguments.
    return subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603


def osascript_picker(start: Path, run: Run = _run) -> Path | None:
    """Show macOS's "choose folder" dialog at ``start``; None if the user cancelled."""
    done = run(["/usr/bin/osascript", "-e", _CHOOSE, PROMPT, str(start)])
    if done.returncode != 0:
        if _CANCELLED in done.stderr:
            return None
        raise DialogError(done.stderr.strip() or f"osascript exited with {done.returncode}")
    # Only the final newline is the script's: a folder name may itself end in one.
    chosen = done.stdout[:-1] if done.stdout.endswith("\n") else done.stdout
    return Path(chosen)


def osascript_alert(message: str, run: Run = _run) -> None:
    run(["/usr/bin/osascript", "-e", _ALERT, message])


def _same(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist
        return False


def refusal(folder: Path, home: Path) -> str | None:
    """Why ``folder`` can't be opened, or None if it can."""
    real = Path(os.path.realpath(folder))
    # samefile, not ==: APFS ignores case, so /Users/ME is the home folder too.
    if _same(real, Path(real.anchor)):
        what = "your whole disk"
    elif _same(real, home):
        what = "your whole home folder"
    else:
        try:
            os.listdir(real)
        except PermissionError:
            return (
                f"askpdf may not read {real.name}. Allow it in System Settings → Privacy &"
                " Security → Files and Folders, then choose the folder again."
            )
        except OSError as exc:
            return f"askpdf can't open {real.name}: {exc.strerror or exc}."
        return None
    return (
        f"askpdf reads every PDF under the folder you choose. That is {what}, which is far"
        " more than askpdf is built for. Choose a folder that holds your PDFs."
    )


def choose_folder(
    pick: Callable[[Path], Path | None],
    alert: Callable[[str], None],
    home: Path,
) -> Path | None:
    """Ask until the person picks a folder askpdf can use, or cancels (None)."""
    while True:
        chosen = pick(home)
        if chosen is None:
            return None
        reason = refusal(chosen, home)
        if reason is None:
            return chosen
        alert(reason)
