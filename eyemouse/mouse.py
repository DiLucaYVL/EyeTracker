"""Windows mouse output (SetCursorPos / SendInput) and physical-click listener."""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from typing import Callable

user32 = ctypes.windll.user32

_LEFT_DOWN, _LEFT_UP, _RIGHT_DOWN, _RIGHT_UP = 0x0002, 0x0004, 0x0008, 0x0010
_LLMHF_INJECTED = 0x01


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _InputUnion)]


def enable_dpi_awareness() -> None:
    """Make coordinates physical pixels; must run before any window is created."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def move_to(x: float, y: float) -> None:
    user32.SetCursorPos(int(round(x)), int(round(y)))


def _send(flags: int) -> None:
    inp = _Input(type=0, u=_InputUnion(mi=_MouseInput(0, 0, 0, flags, 0, 0)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_Input))


def button_down(button: str) -> None:
    _send(_LEFT_DOWN if button == "left" else _RIGHT_DOWN)


def button_up(button: str) -> None:
    _send(_LEFT_UP if button == "left" else _RIGHT_UP)


def click(button: str = "left") -> None:
    button_down(button)
    time.sleep(0.02)
    button_up(button)


def focus_window(win) -> None:
    """Bring a Tk window to the foreground with keyboard focus.

    Windows refuses focus changes from a process that is not in the foreground (e.g. after a global hotkey), so
    briefly attach to the foreground thread's input queue, which makes SetForegroundWindow succeed.
    """
    from ctypes import wintypes

    u32, k32 = user32, ctypes.windll.kernel32
    u32.GetParent.argtypes, u32.GetParent.restype = [wintypes.HWND], wintypes.HWND
    u32.GetForegroundWindow.restype = wintypes.HWND
    u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
    u32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    u32.SetForegroundWindow.argtypes = [wintypes.HWND]
    u32.BringWindowToTop.argtypes = [wintypes.HWND]
    hwnd = u32.GetParent(win.winfo_id()) or win.winfo_id()
    fg = u32.GetForegroundWindow()
    fg_thread = u32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = k32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and u32.AttachThreadInput(me, fg_thread, True))
    try:
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            u32.AttachThreadInput(me, fg_thread, False)


_WHEEL, _HWHEEL = 0x0800, 0x1000


def wheel(vertical: int = 0, horizontal: int = 0) -> None:
    """Scroll by wheel units (120 = one notch). vertical > 0 scrolls up, horizontal > 0 scrolls right."""
    for flag, delta in ((_WHEEL, vertical), (_HWHEEL, horizontal)):
        if delta:
            inp = _Input(type=0, u=_InputUnion(mi=_MouseInput(0, 0, delta & 0xFFFFFFFF, flag, 0, 0)))
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_Input))


def start_click_listener(on_click: Callable[[int, int], None]):
    """Listen to *physical* left clicks (clicks injected by this program are ignored)."""
    from pynput import mouse as pmouse

    def event_filter(msg, data):
        return not (data.flags & _LLMHF_INJECTED)

    def handler(x, y, button, pressed):
        if pressed and button == pmouse.Button.left:
            on_click(int(x), int(y))

    listener = pmouse.Listener(on_click=handler, win32_event_filter=event_filter)
    listener.daemon = True
    listener.start()
    return listener


def start_hotkeys(mapping: dict[str, Callable[[], None]]):
    from pynput import keyboard

    listener = keyboard.GlobalHotKeys(mapping)
    listener.daemon = True
    listener.start()
    return listener
