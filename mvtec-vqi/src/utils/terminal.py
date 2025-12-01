import os
import sys
from contextlib import contextmanager
from typing import List, Optional, Tuple


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


@contextmanager
def _raw_mode(fd):
    if os.name == "nt":
        yield
        return
    import termios
    import tty

    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key() -> str:
    if os.name == "nt":
        import msvcrt

        while True:
            ch = msvcrt.getch()
            if ch in (b"\x03", b"\x1a"):
                raise KeyboardInterrupt
            if ch in (b"\r", b"\n"):
                return "ENTER"
            if ch in (b"\x1b",):
                return "ESC"
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"H":
                    return "UP"
                if ch2 == b"P":
                    return "DOWN"
                if ch2 == b"K":
                    return "LEFT"
                if ch2 == b"M":
                    return "RIGHT"
            if ch:
                return ch.decode(errors="ignore").upper()
    else:
        fd = sys.stdin.fileno()
        with _raw_mode(fd):
            ch1 = sys.stdin.read(1)
            if ch1 == "\x03":
                raise KeyboardInterrupt
            if ch1 in ("\r", "\n"):
                return "ENTER"
            if ch1 == "\x1b":
                sequence = sys.stdin.read(2)
                if sequence == "[A":
                    return "UP"
                if sequence == "[B":
                    return "DOWN"
                if sequence == "[C":
                    return "RIGHT"
                if sequence == "[D":
                    return "LEFT"
                return "ESC"
            return ch1.upper()


def render_menu(
    title: str,
    options: List[Tuple[str, object]],
    selected_index: int,
    current_value: Optional[object] = None,
    footer: Optional[str] = None,
) -> None:
    clear_screen()
    print(title)
    print("Uzyj strzalek gora/dol i klawisza Enter, aby wybrac.")
    print()
    for idx, (label, value) in enumerate(options):
        cursor = "->" if idx == selected_index else "  "
        suffix = " (current)" if current_value is not None and value == current_value else ""
        print(f"{cursor} {label}{suffix}")
    if footer:
        print()
        print(footer)


def select_option(
    title: str,
    options: List[Tuple[str, object]],
    current_value: Optional[object] = None,
    footer: Optional[str] = None,
) -> Optional[object]:
    if not options:
        return None
    index = 0
    while True:
        render_menu(title, options, index, current_value=current_value, footer=footer)
        key = read_key()
        if key == "UP":
            index = (index - 1) % len(options)
        elif key == "DOWN":
            index = (index + 1) % len(options)
        elif key == "ENTER":
            return options[index][1]
        elif key in {"ESC", "Q"}:
            return None
