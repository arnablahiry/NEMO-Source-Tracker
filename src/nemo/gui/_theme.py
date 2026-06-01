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
    mapping = C._DARK_TO_LIGHT if mode == "light" else C._LIGHT_TO_DARK
    C._current_theme = mode

    # Update module-level color constants so newly created widgets use the right colors
    theme = C.LIGHT_THEME if mode == "light" else C.DARK_THEME
    for key in ("BG", "CARD_BG", "CARD_OFF", "ACCENT", "CARD_BORDER", "DIM",
                "DIM_TXT", "LOG_BG", "PLACEHOLDER_TXT", "PLACEHOLDER_BG_EN",
                "PLACEHOLDER_BG_DIS", "STEP_LABEL_TXT", "STEP_LABEL_DIS",
                "BANNER_BTN_TXT", "BANNER_BTN_HOVER"):
        if key in theme:
            setattr(C, key, theme[key])

    _walk(root_widget, mapping)


def _walk(widget, mapping: dict) -> None:
    """Recursively walk widget tree and remap color properties."""
    # Standard tk color properties
    for prop in ("bg", "fg", "highlightbackground", "highlightcolor",
                 "selectcolor", "activebackground", "activeforeground",
                 "troughcolor", "insertbackground", "buttonbackground"):
        try:
            val = widget.cget(prop)
            if val and isinstance(val, str) and val.lower() in mapping:
                widget.configure(**{prop: mapping[val.lower()]})
        except Exception:
            pass

    # _FlatBtn special handling
    try:
        from .widgets import _FlatBtn
        if isinstance(widget, _FlatBtn):
            if widget._bg_on.lower() in mapping:
                widget._bg_on = mapping[widget._bg_on.lower()]
            widget._refresh(mapping)
    except Exception:
        pass

    # Recurse to children
    for child in widget.winfo_children():
        _walk(child, mapping)
