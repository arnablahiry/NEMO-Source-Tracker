"""Theme switching for dark/light mode."""
from . import _constants as C


def apply_theme(root_widget, mode: str) -> None:
    """Recursively remap all widget colors from dark to light or vice versa.

    Parameters
    ----------
    root_widget : tk.Widget
        Top-level widget (typically the root Tk window).
    mode : str
        "dark" or "light".
    """
    # Regenerate mapping dictionaries dynamically to ensure current theme values are used
    mapping = {v.lower(): C.LIGHT_THEME[k] for k, v in C.DARK_THEME.items()} if mode == "light" else {v.lower(): C.DARK_THEME[k] for k, v in C.LIGHT_THEME.items()}
    C._current_theme = mode

    # Update module-level color constants so newly created widgets use the right colors
    theme = C.LIGHT_THEME if mode == "light" else C.DARK_THEME
    for key in ("BG", "CARD_BG", "CARD_OFF", "ACCENT", "ACCENT_HOVER", "CARD_BORDER", "DIM",
                "DIM_TXT", "LOG_BG", "LOG_TXT", "BUTTON_BG", "BUTTON_TXT",
                "PLACEHOLDER_TXT", "PLACEHOLDER_BG_EN", "PLACEHOLDER_BG_DIS",
                "STEP_LABEL_TXT", "STEP_LABEL_DIS",
                "BANNER_BTN_TXT", "BANNER_BTN_HOVER"):
        if key in theme:
            setattr(C, key, theme[key])

    _walk(root_widget, mapping)


def _walk(widget, mapping: dict) -> None:
    """Recursively walk widget tree and remap color properties."""
    # Standard tk color properties
    for prop in ("bg", "fg", "highlightbackground", "highlightcolor",
                 "selectcolor", "activebackground", "activeforeground",
                 "troughcolor", "insertbackground", "buttonbackground",
                 "readonlybackground", "disabledbackground",
                 "disabledforeground"):
        try:
            val = widget.cget(prop)
            if val and isinstance(val, str) and val.lower() in mapping:
                widget.configure(**{prop: mapping[val.lower()]})
        except Exception:
            pass

    # Widgets that manage their own drawing via _refresh()
    if hasattr(widget, '_refresh'):
        try:
            widget._refresh(mapping)
        except Exception:
            pass

    # Recurse to children
    for child in widget.winfo_children():
        _walk(child, mapping)
