"""Shared playback transport: play/pause + loop pills and an FPS box that
animate a host's channel ``tk.Scale``.

Mixed into both the Individual-analysis window and the SliceViewer so the two
share *identical* controls.  The host must provide:

* ``self._slider``      — a ``tk.Scale`` over ``0 … N-1`` whose command redraws.
* ``self._slider_box``  — the accent frame around the slider (for height sync).

Call ``self._init_transport_state()`` before building widgets, then
``self._build_transport(parent)`` to lay out the pills + FPS box (packed
left-to-right into *parent*).  The number of frames is read from the slider's
``to`` value, so the host only ever sets the slider range.
"""
import tkinter as tk

from . import _constants as C


def _accent():
    return C.LOG_TXT if C._current_theme == "light" else C.ACCENT


class TransportControls:
    PILL = 24   # initial pill size; synced to the slider-box height after layout

    # ─────────────────────────── state ──────────────────────────────────────
    def _init_transport_state(self):
        self._anim_playing = False
        self._anim_after   = None
        self._anim_loop    = tk.BooleanVar(value=False)
        self._anim_fps     = tk.IntVar(value=8)
        self._fps_str      = tk.StringVar(value="8")
        self._loop_icons   = {}            # (size, invert) → ImageTk.PhotoImage
        self._pill_size    = self.PILL

    # ─────────────────────────── widgets ────────────────────────────────────
    def _build_transport(self, parent):
        """Play pill, loop pill, then FPS box — packed side-by-side into *parent*."""
        self._play_canvas = tk.Canvas(parent, bg=C.CARD_BG, highlightthickness=0,
                                      bd=0, width=self.PILL, height=self.PILL)
        self._play_canvas.pack(side=tk.LEFT, padx=(6, 0))
        self._play_canvas.bind("<Button-1>", lambda _e: self._anim_toggle())
        self._loop_canvas = tk.Canvas(parent, bg=C.CARD_BG, highlightthickness=0,
                                      bd=0, width=self.PILL, height=self.PILL)
        self._loop_canvas.pack(side=tk.LEFT, padx=(4, 0))
        self._loop_canvas.bind("<Button-1>", lambda _e: self._toggle_loop())
        self._build_fps_control(parent)

        self._draw_play_pill()
        self._draw_loop_pill()
        if getattr(self, "_slider_box", None) is not None:
            self._slider_box.bind("<Configure>",
                                  lambda _e: self._sync_pill_heights())

    def _build_fps_control(self, parent):
        """FPS textbox with borderless square – / + buttons, after the loop pill."""
        rowf = tk.Frame(parent, bg=C.CARD_BG)
        rowf.pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(rowf, text="FPS", bg=C.CARD_BG, fg=C.STEP_LABEL_TXT,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 3))

        def _sq_btn(text, cmd):
            # Label, not Button — no native bezel/shadow/border on any platform.
            lb = tk.Label(rowf, text=text, bg=C.CARD_BG, fg=_accent(),
                          width=2, font=("Helvetica", 12, "bold"), cursor="pointinghand")
            lb.bind("<Button-1>", lambda _e: cmd())
            return lb
        _sq_btn("–", self._anim_slower).pack(side=tk.LEFT)
        ebox = tk.Frame(rowf, bg=_accent(), padx=1, pady=1)
        ebox.pack(side=tk.LEFT, padx=2)
        entry = tk.Entry(ebox, textvariable=self._fps_str, width=3, justify="center",
                         bg=C.CARD_BG, fg=_accent(), insertbackground=_accent(),
                         relief=tk.FLAT, highlightthickness=0,
                         font=("Courier", 9, "bold"))
        entry.pack(fill=tk.BOTH, expand=True, ipady=2)
        entry.bind("<Return>", lambda _e: self._commit_fps())
        entry.bind("<FocusOut>", lambda _e: self._commit_fps())
        _sq_btn("+", self._anim_faster).pack(side=tk.LEFT)

    def _sync_pill_heights(self):
        """Match the play/loop pills exactly to the slider box height."""
        h = self._slider_box.winfo_height()
        if h <= 1 or h == getattr(self, "_pill_size", None):
            return
        self._pill_size = h
        for cv in (self._play_canvas, self._loop_canvas):
            cv.configure(width=h, height=h)
        self._draw_play_pill()
        self._draw_loop_pill()

    # ─────────────────────────── drawing ────────────────────────────────────
    def _draw_play_pill(self):
        cv = self._play_canvas
        cv.delete("all")
        s = getattr(self, "_pill_size", self.PILL)
        light = C._current_theme == "light"
        if self._anim_playing:
            fill = outline = _accent()
            txt = "#ffffff" if light else "#000000"
            icon = "❚❚"
        else:
            fill, outline, txt = C.CARD_BG, C.DIM_TXT, _accent()
            icon = "▶"
        cv.create_rectangle(1, 1, s - 1, s - 1, fill=fill, outline=outline, width=1)
        cv.create_text(s / 2, s / 2, text=icon, anchor="center",
                       font=("Helvetica", 9, "bold"), fill=txt)

    def _loop_icon(self, size, invert):
        """Cached ImageTk loop icon (assets/loop.png); RGB inverted when *invert*."""
        key = (size, invert)
        if key in self._loop_icons:
            return self._loop_icons[key]
        try:
            from PIL import Image, ImageOps, ImageTk
            img = Image.open(C._ASSETS / "loop.png").convert("RGBA")
            img = img.resize((size, size), Image.LANCZOS)
            if invert:
                r, g, b, a = img.split()
                rgb = ImageOps.invert(Image.merge("RGB", (r, g, b)))
                img = Image.merge("RGBA", (*rgb.split(), a))
            photo = ImageTk.PhotoImage(img)
        except Exception:
            photo = None
        self._loop_icons[key] = photo
        return photo

    def _draw_loop_pill(self):
        cv = self._loop_canvas
        cv.delete("all")
        s = getattr(self, "_pill_size", self.PILL)
        on = self._anim_loop.get()
        fill, outline = (_accent(), _accent()) if on else (C.CARD_BG, C.DIM_TXT)
        cv.create_rectangle(1, 1, s - 1, s - 1, fill=fill, outline=outline, width=1)
        pad  = max(5, s // 4)                       # smaller icon inside the square
        icon = self._loop_icon(max(s - 2 * pad, 1), invert=on)  # invert when active
        if icon is not None:
            cv.create_image(s / 2, s / 2, image=icon, anchor="center")
        else:                                       # fallback if PNG unavailable
            cv.create_text(s / 2, s / 2, text="↻", anchor="center",
                           font=("Helvetica", 12, "bold"),
                           fill="#ffffff" if on else C.DIM_TXT)

    def _toggle_loop(self):
        self._anim_loop.set(not self._anim_loop.get())
        self._draw_loop_pill()

    # ─────────────────────────── FPS ────────────────────────────────────────
    def _set_fps(self, v):
        v = max(1, min(int(v), 60))
        self._anim_fps.set(v)
        self._fps_str.set(str(v))
        # apply the new rate immediately while playing (don't wait for the
        # current frame's delay to elapse)
        if self._anim_playing:
            if self._anim_after is not None:
                try:
                    self.after_cancel(self._anim_after)
                except Exception:
                    pass
            self._anim_schedule()

    def _commit_fps(self):
        try:
            self._set_fps(int(float(self._fps_str.get())))
        except (ValueError, TypeError):
            self._fps_str.set(str(int(self._anim_fps.get())))

    def _anim_faster(self):
        self._set_fps(int(self._anim_fps.get()) + 1)

    def _anim_slower(self):
        self._set_fps(int(self._anim_fps.get()) - 1)

    # ─────────────────────── playback transport ─────────────────────────────
    def _anim_nframes(self):
        try:
            return int(float(self._slider.cget("to"))) + 1
        except Exception:
            return 0

    def _anim_toggle(self):
        self._anim_stop() if self._anim_playing else self._anim_start()

    def _anim_start(self):
        if self._anim_nframes() <= 0:
            return
        self._anim_playing = True
        self._draw_play_pill()
        if int(self._slider.get()) >= self._anim_nframes() - 1:
            self._slider.set(0)         # restart if parked at the last frame
        self._anim_schedule()

    def _anim_stop(self):
        self._anim_playing = False
        if self._anim_after is not None:
            try:
                self.after_cancel(self._anim_after)
            except Exception:
                pass
            self._anim_after = None
        if hasattr(self, "_play_canvas"):
            self._draw_play_pill()

    def _anim_schedule(self):
        fps = max(int(self._anim_fps.get()), 1)
        self._anim_after = self.after(int(1000 / fps), self._anim_step)

    def _anim_step(self):
        if not self._anim_playing or not self.winfo_exists():
            return
        last = self._anim_nframes() - 1
        nxt = int(self._slider.get()) + 1
        if nxt > last:
            if self._anim_loop.get():
                nxt = 0
            else:
                self._anim_stop()
                return
        self._slider.set(nxt)           # triggers the slider command (redraw)
        self._anim_schedule()
