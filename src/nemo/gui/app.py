import tkinter as tk
from tkinter import messagebox

from ._constants import (ACCENT, BG, CARD_BG, DIM,
                         BANNER_W, BTN_H, _ASSETS)
from .card import CubeCard


class NemoGUI(tk.Tk):
    N_CARDS = 4
    GIF_INTERVAL_MS = 220

    def __init__(self):
        super().__init__()
        self.title("N.E.M.O : Graphical Interface")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.lift()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))
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
             "Moment 0"),
            ("Wavelet Detections",
             "This step computes the per-spectral channel "
             "unidentified detections of emissions with "
             "2D Starlet transform and thresholding "
             "in wavelet space."),
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

        grid = tk.Frame(self, bg=BG)
        self.cards: list[CubeCard] = []
        for i in range(self.N_CARDS):
            name, desc = _STEPS[i]
            card = CubeCard(grid, index=i, name=name, description=desc,
                            app=self, on_loaded=self._on_card_loaded)
            card.grid(row=0, column=i, padx=6, pady=6)
            self.cards.append(card)

        self.update_idletasks()
        content_h = grid.winfo_reqheight()

        banner = tk.Frame(self, bg=BG, height=content_h)
        banner.pack_propagate(False)
        banner_w = self._build_banner(banner, content_h)
        banner.configure(width=banner_w)
        banner.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)

        tk.Frame(self, bg=DIM, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=10)
        grid.pack(side=tk.LEFT, padx=10, pady=10)

        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    # --- Banner layout knobs ------------------------------------------------
    # >>> ADJUST THIS to move the GitHub / Docs row up or down on the banner.
    #     Value is a fraction of the total banner height (0.0 = top, 1.0 = bottom).
    BANNER_LINKS_TOP_FRAC = 0.24
    def _build_banner(self, frame: tk.Frame, height: int) -> int:
        import webbrowser
        p = _ASSETS / "nemo_vertical.png"
        try:
            from PIL import Image, ImageTk
            try:
                _LANCZOS = Image.Resampling.LANCZOS
            except AttributeError:
                _LANCZOS = Image.LANCZOS

            img = Image.open(p)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")

            disp_h = height
            disp_w = max(1, int(img.width * disp_h / img.height))

            try:
                dpr = max(1, round(float(frame.tk.call('tk', 'scaling')) / 1.3333333))
            except Exception:
                dpr = 1
            phys_w, phys_h = disp_w * dpr, disp_h * dpr

            src_w, src_h = img.size
            while src_h > phys_h * 2:
                img = img.resize((src_w // 2, src_h // 2), _LANCZOS)
                src_w, src_h = img.size
            img = img.resize((phys_w, phys_h), _LANCZOS)

            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, BG)
                bg.paste(img, mask=img.split()[3])
                img = bg

            self._banner_img = ImageTk.PhotoImage(img)
            new_w = disp_w
            tk.Label(frame, image=self._banner_img, bg=BG, bd=0,
                     anchor="n").place(x=0, y=0, width=disp_w, height=disp_h)
        except Exception:
            new_w = BANNER_W
            tk.Label(frame, text="N\nE\nM\nO", bg=BG, fg=ACCENT,
                     font=("Helvetica", 16, "bold")).place(relx=0.5, rely=0.5, anchor="center")

        btn_w = new_w - 8
        btn_h = BTN_H
        gap   = 6
        cx    = new_w // 2

        def _mklink(label, url, x, y, w):
            lbl = tk.Label(
                frame, text=label, bg=BG, fg=ACCENT,
                font=("Helvetica", 10),
                cursor="pointinghand", relief=tk.FLAT,
            )
            lbl.place(x=x, y=y, anchor="n", width=w, height=btn_h)
            lbl.bind("<Button-1>", lambda _e: webbrowser.open(url))
            lbl.bind("<Enter>",    lambda _e: lbl.configure(bg=CARD_BG))
            lbl.bind("<Leave>",    lambda _e: lbl.configure(bg=BG))
            lbl.lift()
            return lbl

        y_links  = int(height * self.BANNER_LINKS_TOP_FRAC)
        half_w   = (btn_w - gap) // 2
        x_left   = 4 + half_w // 2
        x_right  = 4 + half_w + gap + half_w // 2
        _mklink("GitHub", "https://github.com/arnablahiry/NEMO-Source-Tracker",
                x_left,  y_links, half_w)
        _mklink("Docs",   "https://arnablahiry.github.io/software/nemo",
                x_right, y_links, half_w)

        y_rst  = height - btn_h - 6
        y_log  = y_rst  - btn_h - gap

        logs_btn = tk.Label(
            frame, text="View All Logs", bg=BG, fg=ACCENT,
            font=("Helvetica", 10), cursor="pointinghand", relief=tk.FLAT,
        )
        logs_btn.place(x=cx, y=y_log, anchor="n", width=btn_w, height=btn_h)
        logs_btn.bind("<Button-1>", lambda _e: self._view_all_logs())
        logs_btn.bind("<Enter>",    lambda _e: logs_btn.configure(bg=CARD_BG))
        logs_btn.bind("<Leave>",    lambda _e: logs_btn.configure(bg=BG))

        reset_btn = tk.Label(
            frame, text="Reset", bg=BG, fg=ACCENT,
            font=("Helvetica", 10, "bold"), cursor="pointinghand", relief=tk.FLAT,
        )
        reset_btn.place(x=cx, y=y_rst, anchor="n", width=btn_w, height=btn_h)
        reset_btn.bind("<Button-1>", lambda _e: self._reset_pipeline())
        reset_btn.bind("<Enter>",    lambda _e: reset_btn.configure(bg=CARD_BG))
        reset_btn.bind("<Leave>",    lambda _e: reset_btn.configure(bg=BG))

        logs_btn.lift()
        reset_btn.lift()
        return new_w

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
        win.configure(bg=BG)
        win.geometry("700x500")
        win.resizable(True, True)

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(frame, bg=CARD_BG, fg=ACCENT, font=("Courier", 12),
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
        if self.cards[0].cube is not None:
            self.cards[1].enable()
            if len(self.cards) > 2:
                self.cards[2].btn_flow_params.enable()

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
    app = NemoGUI()
    app.mainloop()


if __name__ == "__main__":
    launch()
