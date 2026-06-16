import tkinter as tk
from tkinter import messagebox

from . import _constants as C
from ._constants import (ACCENT,
                         BTN_H,
                         _ASSETS)
from .card import CubeCard


_NATIVE_IV_CLS = None


def _native_imageview_class():
    """Lazily define (exactly once) a click-through NSImageView subclass.

    ObjC classes are registered globally, so defining this twice raises.
    """
    global _NATIVE_IV_CLS
    if _NATIVE_IV_CLS is None:
        import AppKit

        class _NemoBannerImageView(AppKit.NSImageView):
            def hitTest_(self, point):
                return None  # click-through: never intercept mouse events

        _NATIVE_IV_CLS = _NemoBannerImageView
    return _NATIVE_IV_CLS


class NemoGUI(tk.Tk):
    N_CARDS = 4
    GIF_INTERVAL_MS = 220

    def __init__(self):
        super().__init__()
        self.title("N.E.M.O : Graphical Interface")
        self.configure(bg=C.BG)
        self.resizable(False, False)
        self._set_window_appearance()
        self.after(100, self._raise_to_front)
        try:
            from PIL import Image, ImageTk
            _ico = Image.open(_ASSETS / "nemo_ico.png").convert("RGBA")
            self._icon_img = ImageTk.PhotoImage(_ico)
            self.wm_iconphoto(True, self._icon_img)
        except Exception:
            pass

        self._gif_master_chs: list[int] = []
        self._gif_master_idx = 0
        self._gif_job = None

        _STEPS = [
            ("Moment 0",
             "Load and visualize your spectral cube. "
             "The moment 0 map shows integrated emission "
             "across all spectral channels."),
            ("Wavelet Detections",
             "Detect emission sources per channel using 2D Starlet "
             "wavelet decomposition. Supports single-scale or "
             "multi-scale (hierarchical) detection with "
             "configurable thresholding."),
            ("Flow Tracking",
             "TV-L1 masked optical flow is computed between "
             "consecutive channel pairs and used to link "
             "source detections across channels via "
             "Hungarian assignment."),
            ("Source Grouping",
             "Tracks are linked across spectral channels via "
             "Hungarian overlap assignment and grouped into "
             "physical sources via union-find on split and "
             "merge relationships."),
        ]

        # Root layout: vertical banner strip on the left, 2x2 card grid right
        root_row = tk.Frame(self, bg=C.BG)
        root_row.pack(fill=tk.BOTH, expand=True)

        # Outer strip frame: super faint accent border (matches card borders)
        banner_outer = tk.Frame(root_row, bg=C.CARD_BORDER)
        banner_outer.pack(side=tk.LEFT, padx=(10, 0), pady=16)

        grid = tk.Frame(root_row, bg=C.BG)
        self.cards: list[CubeCard] = []
        for i in range(self.N_CARDS):
            name, desc = _STEPS[i]
            card = CubeCard(grid, index=i, name=name, description=desc,
                            app=self, on_loaded=self._on_card_loaded)
            card.grid(row=i // 2, column=i % 2, padx=6, pady=6)
            self.cards.append(card)
        grid.pack(side=tk.LEFT, padx=10, pady=10)

        grid.update_idletasks()
        strip_h = grid.winfo_reqheight() - 12   # visually align with card grid
        strip_w = self._logo_width_for_height(strip_h)

        # Inner strip frame (background color); 1px margin = faint border
        self._banner_frame = tk.Frame(banner_outer, bg=C.BG,
                                      width=strip_w, height=strip_h)
        self._banner_frame.pack_propagate(False)
        self._banner_frame.pack(padx=1, pady=1)

        self._strip_w, self._strip_h = strip_w, strip_h
        self._build_banner(self._banner_frame, strip_w, strip_h)

        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _logo_width_for_height(self, height: int, fallback: int = 200) -> int:
        try:
            from PIL import Image
            img = Image.open(_ASSETS / "nemo_vertical.png")
            return max(fallback, int(img.width * height / img.height))
        except Exception:
            return fallback

    def _build_banner(self, frame: tk.Frame, strip_w: int, strip_h: int):
        import webbrowser
        # Use theme-specific vertical logo
        logo_name = ("nemo_vertical_light.png" if C._current_theme == "light"
                     else "nemo_vertical.png")
        p = _ASSETS / logo_name
        try:
            from PIL import Image, ImageTk, ImageFilter
            try:
                _LANCZOS = Image.Resampling.LANCZOS
            except AttributeError:
                _LANCZOS = Image.LANCZOS

            img = Image.open(p)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")

            # Composite alpha onto the theme background at FULL source
            # resolution (compositing after downsampling causes edge halos)
            if img.mode == "RGBA":
                hex_color = C.BG.lstrip('#')
                bg_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                bg = Image.new("RGB", img.size, bg_rgb)
                bg.paste(img, mask=img.split()[3])
                img = bg

            # Single direct LANCZOS downsample from the full-res source —
            # no intermediate halving, which loses detail — then a mild
            # unsharp mask to keep text and edges crisp at display size
            img = img.resize((strip_w, strip_h), _LANCZOS, reducing_gap=3.0)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))

            self._banner_img = ImageTk.PhotoImage(img)
            # padx/pady=0: Labels default to 1px internal padding, which
            # would offset this image 1pt from the native overlay above it
            tk.Label(frame, image=self._banner_img, bg=C.BG, bd=0,
                     padx=0, pady=0, highlightthickness=0,
                     anchor="nw").place(x=0, y=0, width=strip_w, height=strip_h)
        except Exception:
            tk.Label(frame, text="N.E.M.O", bg=C.BG, fg=ACCENT,
                     font=("Helvetica", 20, "bold"), wraplength=strip_w - 20
                     ).place(relx=0.5, y=30, anchor="n")

        # Canvas-based buttons (SONGS style): accent border + accent text,
        # hover fills solid — 2x2 grid at the bottom of the strip
        BH    = BTN_H
        gap   = 6
        PAD   = 10
        btn_w = (strip_w - 2 * PAD - gap) // 2

        self._banner_cvs: list[tuple] = []  # (canvas, redraw_fn)

        def _make_cv(text, x, y, w, cmd=None, url=None, is_theme=False):
            fnt = ("Courier", 8, "bold") if not is_theme else ("Arial", 11, "bold")
            cv = tk.Canvas(frame, width=w, height=BH,
                           highlightthickness=0, bd=0, cursor="pointinghand")
            cv._lbl_text = text
            cv._lbl_font = fnt
            cv._is_theme = is_theme

            def _redraw(cv=cv, text=text, w=w):
                _t_text = cv._lbl_text  # updated when theme changes
                cv.configure(bg=C.BG)
                cv.delete("all")
                cv.create_rectangle(1, 1, w - 1, BH - 1,
                                    fill=C.BG, outline=C.ACCENT, width=1)
                cv.create_text(w // 2, BH // 2,
                               text=_t_text, fill=C.ACCENT, font=cv._lbl_font)

            def _on_enter(e, cv=cv, w=w):
                cv.delete("all")
                cv.create_rectangle(1, 1, w - 1, BH - 1,
                                    fill=C.ACCENT, outline=C.ACCENT, width=1)
                cv.create_text(w // 2, BH // 2,
                               text=cv._lbl_text, fill=C.BG, font=cv._lbl_font)

            _redraw()
            if url:
                cv.bind("<ButtonRelease-1>", lambda e: webbrowser.open(url))
            elif cmd:
                cv.bind("<ButtonRelease-1>", lambda e: cmd())
            cv.bind("<Enter>", _on_enter)
            cv.bind("<Leave>", lambda e, fn=_redraw: fn())
            cv.place(x=x, y=y, width=w, height=BH)
            self._banner_cvs.append((cv, _redraw))
            return cv

        # Bottom-anchored 2x2 grid: theme | GitHub  /  Docs | Logs
        block_h = 2 * BH + gap
        x0, x1 = PAD, PAD + btn_w + gap
        y0     = strip_h - PAD - block_h
        y1     = y0 + BH + gap
        self._theme_btn = _make_cv(
            "☀" if C._current_theme == "dark" else "☾",
            x0, y0, btn_w, cmd=self._toggle_theme, is_theme=True)
        _make_cv("GitHub", x1, y0, btn_w,
                 url="https://github.com/arnablahiry/NEMO-Source-Tracker")
        _make_cv("Docs",   x0, y1, btn_w,
                 url="https://arnablahiry.github.io/software/nemo")
        _make_cv("Logs",   x1, y1, btn_w, cmd=self._view_all_logs)

        # Retina-sharp native overlay (no-op if PyObjC unavailable); short
        # delay so layout settles — the method retries if the NSWindow
        # hasn't been mapped yet at startup. Reserve the button block at
        # the bottom of the strip so the Tk buttons stay visible.
        reserved_bottom = PAD + block_h + 6
        self.after(50, lambda: self._add_native_banner_overlay(
            p, strip_w, strip_h, reserved_bottom))

    def _raise_to_front(self):
        """Bring the window to the front after the event loop has started."""
        self.lift()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))
        try:
            import AppKit
            AppKit.NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    def _set_window_appearance(self):
        """Force the macOS title bar to match the app theme, overriding the
        system (light/dark) appearance."""
        try:
            import AppKit
            self.update_idletasks()  # make sure the NSWindow exists
            title = str(self.title())
            nswin = next((w for w in AppKit.NSApplication.sharedApplication().windows()
                          if str(w.title()) == title), None)
            if nswin is None:
                return
            name = (AppKit.NSAppearanceNameAqua if C._current_theme == "light"
                    else AppKit.NSAppearanceNameDarkAqua)
            nswin.setAppearance_(AppKit.NSAppearance.appearanceNamed_(name))
        except Exception:
            # Non-macOS / PyObjC unavailable — leave the title bar as-is.
            pass

    def _add_native_banner_overlay(self, logo_path, disp_w, disp_h, reserved_bottom):
        """Overlay a Retina-aware NSImageView over the banner image.

        Tk 8.6 draws PhotoImages point-per-pixel, which macOS stretches 2x
        onto HiDPI screens (blurry). NSImageView renders at the native
        backing scale, so the logo stays crisp. The overlay stops short of
        the button column and is click-through; on any failure the
        Tk-rendered banner simply remains visible as the fallback.
        """
        try:
            import os
            import tempfile
            import AppKit
            from PIL import Image

            title = self.title()
            nswin = next((w for w in AppKit.NSApplication.sharedApplication().windows()
                          if str(w.title()) == title), None)
            if nswin is None:
                # window not mapped yet (startup) — retry once shortly
                if not getattr(self, '_banner_overlay_retried', False):
                    self._banner_overlay_retried = True
                    self.after(300, lambda: self._add_native_banner_overlay(
                        logo_path, disp_w, disp_h, reserved_bottom))
                return
            self._banner_overlay_retried = False
            content = nswin.contentView()

            if getattr(self, "_banner_overlay_view", None) is not None:
                self._banner_overlay_view.removeFromSuperview()
                self._banner_overlay_view = None

            # banner-frame position within the window (Tk: top-left origin)
            self.update_idletasks()
            fx = self._banner_frame.winfo_rootx() - self.winfo_rootx()
            fy = self._banner_frame.winfo_rooty() - self.winfo_rooty()

            # stop above the bottom button block so Tk buttons stay visible
            overlay_w = disp_w
            overlay_h = max(1, disp_h - reserved_bottom)

            # crop the matching top fraction of the FULL-RES source
            src = Image.open(logo_path)
            crop_px = max(1, round(src.height * overlay_h / disp_h))
            tmp = os.path.join(tempfile.gettempdir(), "_nemo_banner_overlay.png")
            src.crop((0, 0, src.width, crop_px)).save(tmp)
            nsimg = AppKit.NSImage.alloc().initWithContentsOfFile_(tmp)
            if nsimg is None:
                return

            if content.isFlipped():
                oy = fy
            else:
                oy = content.frame().size.height - fy - overlay_h
            iv = _native_imageview_class().alloc().initWithFrame_(
                ((fx, oy), (overlay_w, overlay_h)))
            iv.setImage_(nsimg)
            iv.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
            iv.setImageFrameStyle_(AppKit.NSImageFrameNone)
            content.addSubview_(iv)
            self._banner_overlay_view = iv
        except Exception:
            pass

    def _reload_banner(self):
        try:
            # drop the stale native overlay immediately so the old theme's
            # logo never lingers on top while widgets beneath are re-themed
            if getattr(self, '_banner_overlay_view', None) is not None:
                try:
                    self._banner_overlay_view.removeFromSuperview()
                except Exception:
                    pass
                self._banner_overlay_view = None
            if hasattr(self, '_banner_img'):
                del self._banner_img
            if hasattr(self, '_banner_cvs'):
                self._banner_cvs.clear()
            for widget in self._banner_frame.winfo_children():
                widget.destroy()
            self._build_banner(self._banner_frame, self._strip_w, self._strip_h)
        except Exception:
            pass

    def _disable_theme_button(self):
        cv = self._theme_btn
        dim = C.DIM
        cv.delete("all")
        w, h = cv.winfo_reqwidth(), cv.winfo_reqheight()
        cv.create_rectangle(1, 1, w - 1, h - 1, fill=C.BG, outline=dim, width=1)
        cv.create_text(w // 2, h // 2, text=cv._lbl_text, fill=dim, font=cv._lbl_font)
        cv.unbind("<ButtonRelease-1>")
        cv.unbind("<Enter>")
        cv.unbind("<Leave>")
        cv.configure(cursor="arrow")

    def _enable_theme_button(self):
        cv = self._theme_btn
        w, h = cv.winfo_reqwidth(), cv.winfo_reqheight()
        cv.delete("all")
        cv.create_rectangle(1, 1, w - 1, h - 1, fill=C.BG, outline=C.ACCENT, width=1)
        cv.create_text(w // 2, h // 2, text=cv._lbl_text, fill=C.ACCENT, font=cv._lbl_font)
        cv.configure(cursor="pointinghand")
        cv.bind("<ButtonRelease-1>", lambda e: self._toggle_theme())
        cv.bind("<Enter>", lambda e: (cv.delete("all"),
                                      cv.create_rectangle(1, 1, w-1, h-1, fill=C.ACCENT, outline=C.ACCENT),
                                      cv.create_text(w//2, h//2, text=cv._lbl_text, fill=C.BG, font=cv._lbl_font)))
        cv.bind("<Leave>", lambda e: (cv.delete("all"),
                                      cv.create_rectangle(1, 1, w-1, h-1, fill=C.BG, outline=C.ACCENT),
                                      cv.create_text(w//2, h//2, text=cv._lbl_text, fill=C.ACCENT, font=cv._lbl_font)))

    def _toggle_theme(self):
        from . import _constants as C
        from ._theme import apply_theme
        new = "light" if C._current_theme == "dark" else "dark"
        apply_theme(self, new)
        self._set_window_appearance()
        self._reload_banner()
        for card in self.cards:
            card.refresh_on_theme_change()

    def _view_all_logs(self):
        all_sections = []
        for card in self.cards:
            if card._log_lines:
                all_sections.append(f"\n── {card.name} ──\n")
                all_sections.extend(card._log_lines)

        if not all_sections:
            messagebox.showinfo("No Logs", "No pipeline logs yet. Run the pipeline first.")
            return

        win = tk.Toplevel(self)
        win.title("All Pipeline Logs")
        win.configure(bg=C.BG)
        win.geometry("700x500")
        win.resizable(True, True)

        frame = tk.Frame(win, bg=C.BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, bg=C.CARD_BG, fg=ACCENT, font=("Courier", 12),
                      wrap=tk.WORD, state=tk.DISABLED,
                      yscrollcommand=sb.set, relief=tk.FLAT, bd=0)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.configure(command=txt.yview)
        txt.tag_configure("header", foreground="white", font=("Courier", 12, "bold"))

        txt.configure(state=tk.NORMAL)
        for line in all_sections:
            if line.startswith("\n──"):
                txt.insert(tk.END, line, "header")
            else:
                txt.insert(tk.END, line)
        txt.see(tk.END)
        txt.configure(state=tk.DISABLED)

    def _reset_pipeline(self):
        if not messagebox.askyesno("Reset", "Reset all pipeline results?"):
            return
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        self._gif_master_chs = []
        self._gif_master_idx = 0
        for card in self.cards:
            card.reset()
        # Clear cube from card 0 completely on reset
        self.cards[0].cube = None
        self.cards[0].cube_raw = None
        self.cards[0]._clear_preview()
        self.cards[0]._draw_placeholder()
        # Disable card 0 view button (only Load should be enabled)
        self.cards[0].btn_view.disable()
        # Disable all card 1 buttons that depend on card 0 data
        self.cards[1].btn_decompose.disable()
        self.cards[1].btn_configure.disable()
        self.cards[1].btn_run.disable()
        self.cards[1].btn_det_view.disable()
        # Disable all card 2 controls
        if len(self.cards) > 2:
            self.cards[2]._set_flow_params_enabled(False)
            self.cards[2].btn_flow_view.disable()
        # Disable all card 3 buttons
        if len(self.cards) > 3:
            self.cards[3].btn_view_sources.disable()
            self.cards[3].btn_combined.disable()
            self.cards[3].btn_individual.disable()
        # Re-enable theme button on reset
        self._enable_theme_button()

    def current_gif_channel(self):
        if not self._gif_master_chs:
            return None
        return self._gif_master_chs[self._gif_master_idx]

    def refresh_gif_clock(self):
        union = set()
        for card in self.cards:
            union.update(card._gif_frames_by_ch.keys())
        new_chs = sorted(union)
        if not new_chs:
            return
        prev_ch = self.current_gif_channel()
        self._gif_master_chs = new_chs
        if prev_ch in new_chs:
            self._gif_master_idx = new_chs.index(prev_ch)
        else:
            self._gif_master_idx = 0
        for card in self.cards:
            card.show_gif_for_channel(new_chs[self._gif_master_idx])
        if self._gif_job is None:
            self._tick_gif()

    def _tick_gif(self):
        if not self._gif_master_chs:
            self._gif_job = None
            return
        ch = self._gif_master_chs[self._gif_master_idx]
        for card in self.cards:
            card.show_gif_for_channel(ch)
        self._gif_master_idx = (self._gif_master_idx + 1) % len(self._gif_master_chs)
        self._gif_job = self.after(self.GIF_INTERVAL_MS, self._tick_gif)

    def _on_card_loaded(self, index: int):
        nxt = index + 1
        if nxt < self.N_CARDS:
            self.cards[nxt].enable()


def launch():
    import argparse
    from ._theme import apply_theme
    p = argparse.ArgumentParser(prog="nemo-gui", add_help=False)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--light", action="store_true", help="Launch in light mode")
    g.add_argument("--dark",  action="store_true", help="Launch in dark mode (default)")
    args, _ = p.parse_known_args()

    app = NemoGUI()
    if args.light:
        apply_theme(app, "light")
        app._set_window_appearance()
        app._reload_banner()
        for card in app.cards:
            card.refresh_on_theme_change()
    app.mainloop()


if __name__ == "__main__":
    launch()
