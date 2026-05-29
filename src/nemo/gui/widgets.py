import queue
import tkinter as tk

from ._constants import ACCENT, CARD_OFF, DIM_TXT


class _FlatBtn(tk.Frame):
    """Reliably coloured flat button — tk.Button ignores bg on macOS."""

    def __init__(self, parent, text: str, command,
                 bg_on: str, fg_on: str = "black",
                 font=("Helvetica", 11),
                 active: bool = False,
                 height: int | None = None,
                 btn_width: int | None = None,
                 **kw):
        from ._constants import BTN_W, BTN_H
        self._bg_on  = bg_on
        self._fg_on  = fg_on
        self._cmd    = command
        self._active = False

        h  = height    if height    is not None else BTN_H
        w  = btn_width if btn_width is not None else BTN_W
        bg = bg_on if active else CARD_OFF
        fg = fg_on if active else DIM_TXT

        super().__init__(parent, width=w, height=h,
                         bg=bg, cursor="arrow", **kw)
        self.pack_propagate(False)

        self._lbl = tk.Label(self, text=text, bg=bg, fg=fg,
                             font=font, anchor="center", justify=tk.CENTER)
        self._lbl.place(relwidth=1, relheight=1)

        if active:
            self.enable()

    def enable(self, bg_on: str | None = None):
        if bg_on:
            self._bg_on = bg_on
        self._active = True
        self.configure(bg=self._bg_on, cursor="pointinghand")
        self._lbl.configure(bg=self._bg_on, fg=self._fg_on)
        for w in (self, self._lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>",    self._hover)
            w.bind("<Leave>",    self._leave)

    def disable(self):
        self._active = False
        self.configure(bg=CARD_OFF, cursor="arrow")
        self._lbl.configure(bg=CARD_OFF, fg=DIM_TXT)
        for w in (self, self._lbl):
            w.unbind("<Button-1>")
            w.unbind("<Enter>")
            w.unbind("<Leave>")

    def _click(self, _e=None):
        if self._active and self._cmd:
            self._cmd()

    def _hover(self, _e=None):
        r = int(self._bg_on[1:3], 16)
        g = int(self._bg_on[3:5], 16)
        b = int(self._bg_on[5:7], 16)
        dim = f"#{int(r*.82):02x}{int(g*.82):02x}{int(b*.82):02x}"
        self.configure(bg=dim)
        self._lbl.configure(bg=dim)

    def _leave(self, _e=None):
        self.configure(bg=self._bg_on)
        self._lbl.configure(bg=self._bg_on)


class _QueueStream:
    """File-like object that forwards write() calls into a queue."""
    def __init__(self, q: queue.Queue):
        self._q = q
    def write(self, text: str):
        if text:
            self._q.put(("log", text))
    def flush(self):
        pass
