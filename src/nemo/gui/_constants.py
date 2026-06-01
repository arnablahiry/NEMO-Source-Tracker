from pathlib import Path

BG          = "#1a1a2e"
CARD_BG     = "#16213e"
CARD_OFF    = "#0f0f1a"
ACCENT      = "#4ecca3"
CARD_BORDER = "#1a6060"
DIM         = "#44445a"
DIM_TXT     = "#555577"
LOG_BG      = "#0a0a14"
PLACEHOLDER_TXT = "#666699"
PLACEHOLDER_BG_EN = "#2a2a4a"
PLACEHOLDER_BG_DIS = "#111128"
STEP_LABEL_TXT = "#b0b5d0"
STEP_LABEL_DIS = "#555577"
BANNER_BTN_TXT = "#e0e0e8"
BANNER_BTN_HOVER = "#2a2a4a"
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

DARK_THEME = {
    "BG": "#1a1a2e", "CARD_BG": "#16213e", "CARD_OFF": "#0f0f1a",
    "ACCENT": "#4ecca3", "CARD_BORDER": "#1a6060", "DIM": "#44445a",
    "DIM_TXT": "#555577", "RUN_COLOR": "#f5a623",
    "LOG_BG": "#0a0a14", "PLACEHOLDER_TXT": "#666699",
    "PLACEHOLDER_BG_EN": "#2a2a4a", "PLACEHOLDER_BG_DIS": "#111128",
    "STEP_LABEL_TXT": "#b0b5d0", "STEP_LABEL_DIS": "#555577",
    "BANNER_BTN_TXT": "#e0e0e8", "BANNER_BTN_HOVER": "#2a2a4a",
}

LIGHT_THEME = {
    "BG": "#f5f5fa", "CARD_BG": "#ffffff", "CARD_OFF": "#efeffa",
    "ACCENT": "#abf4e3", "CARD_BORDER": "#2a9090", "DIM": "#d5d5e0",
    "DIM_TXT": "#7a7a9a", "RUN_COLOR": "#c47800",
    "LOG_BG": "#fafafd", "PLACEHOLDER_TXT": "#9999bb",
    "PLACEHOLDER_BG_EN": "#e8e8f5", "PLACEHOLDER_BG_DIS": "#f5f5fa",
    "STEP_LABEL_TXT": "#6a6a8a", "STEP_LABEL_DIS": "#aaaacc",
    "BANNER_BTN_TXT": "#2a2a4a", "BANNER_BTN_HOVER": "#e8e8f0",
}

_DARK_TO_LIGHT = {v.lower(): LIGHT_THEME[k] for k, v in DARK_THEME.items()}
_LIGHT_TO_DARK = {v.lower(): DARK_THEME[k] for k, v in LIGHT_THEME.items()}

_current_theme = "dark"
