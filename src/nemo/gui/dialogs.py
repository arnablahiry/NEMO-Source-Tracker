import tkinter as tk
from tkinter import messagebox

from ._constants import ACCENT, BG, CARD_BG, DIM
from .widgets import _FlatBtn


class WaveletParamsDialog(tk.Toplevel):
    """Modal dialog: edit and save WaveletDetector parameters."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Wavelet Detection — Parameters")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        defaults = dict(scales=6, k_sigma=5.0, use_scale=5,
                        min_area=20, thresh="", use_mean_map_sigma=True)
        if current:
            defaults.update({k: ("" if v is None else v) for k, v in current.items()})

        pad = dict(padx=14, pady=5, sticky="w")
        tk.Label(self, text="Starlet Wavelet Detector", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 8), padx=14)

        fields = [
            ("Scales (total starlet levels)",        "scales",             "int",   (2, 12)),
            ("k-sigma (detection threshold)",        "k_sigma",            "float", None),
            ("Use scale (1-based detail band)",      "use_scale",          "int",   (1, 11)),
            ("Min area (px, discard smaller blobs)", "min_area",           "int",   (1, 500)),
            ("Absolute threshold (blank = auto)",    "thresh",             "str",   None),
            ("Use mean-map sigma reference",         "use_mean_map_sigma", "bool",  None),
        ]
        self._vars: dict[str, tk.Variable] = {}
        for r, (label, key, typ, rng) in enumerate(fields, start=1):
            tk.Label(self, text=label, bg=BG, fg="white",
                     font=("Helvetica", 9), anchor="w").grid(row=r, column=0, **pad)
            if typ == "bool":
                v = tk.BooleanVar(value=bool(defaults[key]))
                tk.Checkbutton(self, variable=v, bg=BG, activebackground=BG,
                               selectcolor=CARD_BG, fg=ACCENT,
                               relief=tk.FLAT, bd=0).grid(
                                   row=r, column=1, padx=14, pady=5, sticky="w")
            elif typ == "int" and rng:
                v = tk.IntVar(value=int(defaults[key]))
                tk.Spinbox(self, from_=rng[0], to=rng[1], textvariable=v,
                           width=6, bg=CARD_BG, fg="white",
                           buttonbackground=CARD_BG,
                           relief=tk.FLAT, insertbackground="white").grid(
                               row=r, column=1, padx=14, pady=5, sticky="w")
            else:
                v = tk.StringVar(value=str(defaults[key]))
                tk.Entry(self, textvariable=v, width=10, bg=CARD_BG, fg="white",
                         insertbackground="white", relief=tk.FLAT).grid(
                             row=r, column=1, padx=14, pady=5, sticky="w")
            self._vars[key] = v

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2, pady=(12, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Save",   self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        def _f(k): return self._vars[k].get()
        thresh_s = str(_f("thresh")).strip()
        try:
            params = dict(
                scales=int(_f("scales")), k_sigma=float(_f("k_sigma")),
                use_scale=int(_f("use_scale")), min_area=int(_f("min_area")),
                thresh=float(thresh_s) if thresh_s else None,
                use_mean_map_sigma=bool(_f("use_mean_map_sigma")),
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()


class FlowParamsDialog(tk.Toplevel):
    """Modal dialog: edit and save FlowTracker parameters."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Flow Tracking — Parameters")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        defaults = dict(min_match_overlap=5, max_gap_channels=5)
        if current:
            defaults.update(current)

        pad = dict(padx=14, pady=5, sticky="w")
        tk.Label(self, text="TV-L1 Flow Tracker", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 8), padx=14)

        fields = [
            ("Min match overlap (px)",   "min_match_overlap", "int", (1, 100)),
            ("Max gap channels",         "max_gap_channels",  "int", (1, 50)),
        ]
        self._vars: dict[str, tk.Variable] = {}
        for r, (label, key, typ, rng) in enumerate(fields, start=1):
            tk.Label(self, text=label, bg=BG, fg="white",
                     font=("Helvetica", 9), anchor="w").grid(row=r, column=0, **pad)
            if typ == "int" and rng:
                v = tk.IntVar(value=int(defaults[key]))
                tk.Spinbox(self, from_=rng[0], to=rng[1], textvariable=v,
                           width=6, bg=CARD_BG, fg="white",
                           buttonbackground=CARD_BG,
                           relief=tk.FLAT, insertbackground="white").grid(
                               row=r, column=1, padx=14, pady=5, sticky="w")
            else:
                v = tk.StringVar(value=str(defaults[key]))
                tk.Entry(self, textvariable=v, width=10, bg=CARD_BG, fg="white",
                         insertbackground="white", relief=tk.FLAT).grid(
                             row=r, column=1, padx=14, pady=5, sticky="w")
            self._vars[key] = v

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=len(fields)+1, column=0, columnspan=2, pady=(12, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Save",   self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        def _f(k): return self._vars[k].get()
        try:
            params = dict(
                min_match_overlap=int(_f("min_match_overlap")),
                max_gap_channels=int(_f("max_gap_channels")),
            )
        except ValueError as exc:
            messagebox.showerror("Bad parameter", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()


class ScalingDialog(tk.Toplevel):
    """Choose Linear / Log / Power scaling for the cube."""

    def __init__(self, master, on_save, current: dict | None = None):
        super().__init__(master)
        self.title("Cube Scaling")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save

        cur_mode  = (current or {}).get("mode", "linear")
        cur_gamma = float((current or {}).get("gamma", 0.5))

        self._mode  = tk.StringVar(value=cur_mode)
        self._gamma = tk.StringVar(value=f"{cur_gamma:g}")

        tk.Label(self, text="Cube Scaling", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).grid(
                     row=0, column=0, columnspan=2, pady=(14, 6), padx=14)
        tk.Label(self, text="Applies to the whole cube. Downstream\n"
                            "detections use the rescaled values.",
                 bg=BG, fg="white", font=("Helvetica", 8),
                 justify=tk.LEFT).grid(row=1, column=0, columnspan=2,
                                       padx=14, pady=(0, 8), sticky="w")

        for r, label in enumerate(("linear", "log", "power")):
            tk.Radiobutton(self, text=label.capitalize(),
                           variable=self._mode, value=label,
                           bg=BG, fg="white", selectcolor=CARD_BG,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Helvetica", 10), relief=tk.FLAT,
                           anchor="w").grid(row=2+r, column=0,
                                            padx=(20, 8), pady=2, sticky="w")

        tk.Label(self, text="gamma", bg=BG, fg="white",
                 font=("Helvetica", 9)).grid(row=4, column=1, padx=(0, 4),
                                              sticky="e")
        tk.Entry(self, textvariable=self._gamma, width=6,
                 bg=CARD_BG, fg="white", font=("Helvetica", 9),
                 insertbackground="white",
                 relief=tk.FLAT).grid(row=4, column=1, padx=(60, 14), sticky="e")

        btn_row = tk.Frame(self, bg=BG)
        btn_row.grid(row=8, column=0, columnspan=2, pady=(14, 14))
        _FlatBtn(btn_row, "Cancel", self.destroy, bg_on=DIM,    active=True).pack(side=tk.LEFT, padx=6)
        _FlatBtn(btn_row, "Apply",  self._save,   bg_on=ACCENT, active=True).pack(side=tk.LEFT, padx=6)

        self.transient(master)
        self.wait_visibility()
        self.update_idletasks()
        px = master.winfo_rootx() + (master.winfo_width()  - self.winfo_width())  // 2
        py = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _save(self):
        try:
            params = dict(mode=self._mode.get(),
                          gamma=float(self._gamma.get()))
        except ValueError as exc:
            messagebox.showerror("Bad gamma", str(exc), parent=self)
            return
        self._on_save(params)
        self.destroy()
