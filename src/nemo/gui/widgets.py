import queue
import tkinter as tk

from . import _constants as C
from ._constants import ACCENT


class _FlatBtn(tk.Frame):
    """Canvas-based button: accent border + accent text, hover fills solid (SONGS style)."""

    def __init__(self, parent, text: str, command,
                 bg_on: str = "",          # unused, kept for call-site compatibility
                 fg_on: str = "",          # unused
                 font=("Arial", 10, "bold"),
                 active: bool = False,
                 height: int | None = None,
                 btn_width: int | None = None,
                 special_color: str = "",
                 **kw):
        from ._constants import BTN_W, BTN_H
        self._cmd    = command
        self._active = False
        self._text   = text
        self._font   = font
        self._special_color = special_color

        h = height    if height    is not None else BTN_H
        w = btn_width if btn_width is not None else BTN_W

        super().__init__(parent, width=w, height=h,
                         bg=C.CARD_BG, cursor="arrow", **kw)
        self.pack_propagate(False)

        self._cv = tk.Canvas(self, highlightthickness=0, bd=0)
        self._cv.place(relwidth=1, relheight=1)

        self._draw()
        if active:
            self.enable()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _draw(self, hover=False):
        cv = self._cv
        cv.delete("all")
        w = int(self.cget("width"))
        h = int(self.cget("height"))

        # Special colors for specific button types
        color_map_light = {
            "red": {"idle": "#c41e3a", "hover": "#8b0000", "bg": "#fff5f5"},
            "yellow": {"idle": "#8b6914", "hover": "#6b5410", "bg": "#fffef0"},
        }
        color_map_dark = {
            "red": {"idle": "#ff6b6b", "hover": "#e74c3c", "bg": "#1a0a0a"},
            "yellow": {"idle": "#d4af37", "hover": "#c9a961", "bg": "#1a1508"},
        }

        color_map = color_map_light if C._current_theme == "light" else color_map_dark
        colors = color_map.get(self._special_color, {})

        if colors:
            # Special styling for both light and dark modes
            if self._active:
                if hover:
                    fill, outline, fg = colors.get("hover", C.ACCENT), colors.get("hover", C.ACCENT), C.BG
                else:
                    fill, outline, fg = colors.get("bg", C.CARD_BG), colors.get("idle", C.ACCENT), colors.get("idle", C.ACCENT)
            else:
                fill, outline, fg = C.BUTTON_BG, C.BUTTON_TXT, C.BUTTON_TXT
        else:
            # Standard styling (no special color)
            if self._active:
                if hover:
                    fill, outline, fg = C.ACCENT, C.ACCENT, C.BG
                else:
                    fill, outline, fg = C.CARD_BG, C.ACCENT, C.ACCENT
            else:
                fill = outline = fg = C.BUTTON_BG
                fg = C.BUTTON_TXT
                outline = C.BUTTON_TXT

        cv.configure(bg=fill)
        cv.create_rectangle(1, 1, w - 1, h - 1, fill=fill, outline=outline, width=1)
        cv.create_text(w // 2, h // 2, text=self._text, fill=fg,
                       font=self._font, justify="center")

    # ── public API ───────────────────────────────────────────────────────────

    def enable(self, bg_on: str | None = None):
        self._active = True
        self.configure(cursor="pointinghand")
        self._draw()
        for w in (self, self._cv):
            # Fire on release (not press) so a press can be cancelled by
            # dragging off the button before letting go.
            w.bind("<ButtonRelease-1>", self._click)
            w.bind("<Enter>",    lambda _e: self._draw(hover=True))
            w.bind("<Leave>",    lambda _e: self._draw(hover=False))

    def disable(self):
        self._active = False
        self.configure(cursor="arrow")
        self._draw()
        for w in (self, self._cv):
            w.unbind("<ButtonRelease-1>")
            w.unbind("<Enter>")
            w.unbind("<Leave>")

    def _refresh(self, mapping: dict | None = None):
        self.configure(bg=C.CARD_BG)
        self._draw()

    def _click(self, e=None):
        if not (self._active and self._cmd):
            return
        # Only trigger if the pointer is still over the button on release.
        if e is not None:
            w, h = self.winfo_width(), self.winfo_height()
            if not (0 <= e.x <= w and 0 <= e.y <= h):
                return
        self._cmd()


def make_slider_box(parent, lo, hi, res, kind: str, value,
                    on_change=None, height: int | None = None,
                    divider_padx=(3, 0)):
    """Accent-bordered slider with a divider and an editable value entry.

    Shared look for every tweakable slider (gamma, optical-flow params,
    false-detection params). Colours are read from the *current* theme.

    Returns a dict with: outer, box, scale, entry, var, get(), set(v), refresh().
    `on_change(value)` fires whenever the value changes (slider or typed entry).
    """
    def _fmt(v):
        return str(int(round(float(v)))) if kind == "int" else f"{float(v):.2f}"

    def _clamp(v):
        v = max(lo, min(hi, v))
        return int(round(v)) if kind == "int" else v

    state = {"v": _clamp(float(value))}

    outer = tk.Frame(parent, bg=C.DIM, padx=1, pady=1)
    if height is not None:
        outer.configure(height=height)
        outer.pack_propagate(False)
    box = tk.Frame(outer, bg=C.ACCENT, padx=1, pady=1)          # accent border
    box.pack(fill=tk.BOTH, expand=True)
    row = tk.Frame(box, bg=C.CARD_BG, padx=4, pady=1)
    row.pack(fill=tk.BOTH, expand=True)

    var = tk.StringVar(value=_fmt(state["v"]))
    scale = tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     bg=C.ACCENT, fg=C.ACCENT, troughcolor=C.LOG_BG,
                     activebackground=C.ACCENT_HOVER, highlightthickness=0,
                     sliderrelief=tk.FLAT, bd=0, showvalue=False, width=8,
                     length=46)   # small request so the value entry keeps its width
    scale.set(state["v"])
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    divider = tk.Frame(row, bg=C.ACCENT, width=1)
    divider.pack(side=tk.LEFT, fill=tk.Y, padx=divider_padx)
    entry = tk.Entry(row, textvariable=var, width=5, justify="right",
                     bg=C.CARD_BG, fg=C.ACCENT, insertbackground=C.ACCENT,
                     disabledbackground=C.CARD_BG, disabledforeground=C.DIM_TXT,
                     relief=tk.FLAT, highlightthickness=0, font=("Courier", 8), bd=0)
    entry.pack(side=tk.RIGHT, padx=(2, 1))

    _guard = {"busy": False}

    def _commit(v, push_scale):
        state["v"] = _clamp(v)
        var.set(_fmt(state["v"]))
        if push_scale:
            _guard["busy"] = True
            scale.set(state["v"])
            _guard["busy"] = False
        if on_change:
            on_change(state["v"])

    def _from_scale(_=None):
        if _guard["busy"]:
            return
        _commit(float(scale.get()), push_scale=False)

    def _from_entry(_=None):
        try:
            _commit(float(var.get()), push_scale=True)
        except ValueError:
            var.set(_fmt(state["v"]))

    scale.configure(command=_from_scale)
    entry.bind("<Return>", _from_entry)
    entry.bind("<FocusOut>", _from_entry)

    def _set(v):
        _commit(float(v), push_scale=True)

    return dict(outer=outer, box=box, row=row, scale=scale, entry=entry,
                divider=divider, var=var, get=lambda: state["v"], set=_set, fmt=_fmt)


class _QueueStream:
    """File-like object that forwards write() calls into a queue."""
    def __init__(self, q: queue.Queue):
        self._q = q
    def write(self, text: str):
        if text:
            self._q.put(("log", text))
    def flush(self):
        pass
