from pathlib import Path

BG          = "#f5f5fa"
CARD_BG     = "#ffffff"
CARD_OFF    = "#f0f0f5"
ACCENT      = "#1a3a7a"  # Active button color — dark blue, must match LIGHT_THEME["ACCENT"]
ACCENT_HOVER = "#2a4f9e"  # Slider marker hover — slightly lighter than ACCENT
CARD_BORDER = "#ccd6ea"  # Super faint accent (light blue) — must match LIGHT_THEME
DIM         = "#e8e8f0"
DIM_TXT     = "#b0b0c0"
LOG_BG      = "#fafafd"
LOG_TXT     = "#1a3a7b"  # Log text color — must match LIGHT_THEME["LOG_TXT"]
BUTTON_BG   = "#f0f0f6"  # Disabled button background — must match LIGHT_THEME
BUTTON_TXT  = "#dcdce0"  # Disabled button text — must match LIGHT_THEME
PLACEHOLDER_TXT = "#9999bb"
PLACEHOLDER_BG_EN = "#e3e6f0"
PLACEHOLDER_BG_DIS = "#f5f5fb"
STEP_LABEL_TXT = "#6a6a8a"
STEP_LABEL_DIS = "#aaaacc"
BANNER_BTN_TXT = "#2a2a4a"
BANNER_BTN_HOVER = "#e8e8f1"
CARD_W      = 270
CARD_H      = 270
BANNER_W    = 80
RUN_COLOR   = "#f5a623"
BTN_W       = 126
BTN_H       = 30
BTN_ZONE_H  = 120
BTN_TALL    = 54

_ASSETS = Path(__file__).parent.parent.parent.parent / "assets"

_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis",
          "gray", "hot", "afmhot", "YlOrRd", "cubehelix"]

# NOTE: theme switching remaps colors BY VALUE, so every value within each
# theme dict must be unique (hence the ±1 hex nudges on near-identical greys).
DARK_THEME = {
    "BG": "#0a0a0f", "CARD_BG": "#1a1a25", "CARD_OFF": "#151519",
    "ACCENT": "#90caf9", "ACCENT_HOVER": "#aad4fb", "CARD_BORDER": "#1d3650", "DIM": "#2a2a35",  # border = super faint accent
    "DIM_TXT": "#4a4a5a", "RUN_COLOR": "#f5a623",
    "LOG_BG": "#000000", "LOG_TXT": "#90caf8",  # Log text color — visually matches ACCENT
    "BUTTON_BG": "#080809", "BUTTON_TXT": "#2a2a36",  # Heavily faded for disabled cards
    "PLACEHOLDER_TXT": "#5a5a7a",
    "PLACEHOLDER_BG_EN": "#131b26", "PLACEHOLDER_BG_DIS": "#0a0a10",
    "STEP_LABEL_TXT": "#a0a0c0", "STEP_LABEL_DIS": "#4a4a5b",
    "BANNER_BTN_TXT": "#d0d0e0", "BANNER_BTN_HOVER": "#1a1a26",
}

LIGHT_THEME = {
    "BG": "#f5f5fa", "CARD_BG": "#ffffff", "CARD_OFF": "#f0f0f5",
    "ACCENT": "#1a3a7a", "ACCENT_HOVER": "#2a4f9e", "CARD_BORDER": "#ccd6ea", "DIM": "#e8e8f0",  # border = super faint accent
    "DIM_TXT": "#b0b0c0", "RUN_COLOR": "#ffdba0",
    "LOG_BG": "#fafafd", "LOG_TXT": "#1a3a7b",  # visually matches light ACCENT
    "BUTTON_BG": "#f0f0f6", "BUTTON_TXT": "#dcdce0",  # Heavily faded for disabled cards
    "PLACEHOLDER_TXT": "#9999bb",
    "PLACEHOLDER_BG_EN": "#e3e6f0", "PLACEHOLDER_BG_DIS": "#f5f5fb",
    "STEP_LABEL_TXT": "#6a6a8a", "STEP_LABEL_DIS": "#aaaacc",
    "BANNER_BTN_TXT": "#2a2a4a", "BANNER_BTN_HOVER": "#e8e8f1",
}

_DARK_TO_LIGHT = {v.lower(): LIGHT_THEME[k] for k, v in DARK_THEME.items()}
_LIGHT_TO_DARK = {v.lower(): DARK_THEME[k] for k, v in LIGHT_THEME.items()}

_current_theme = "light"
