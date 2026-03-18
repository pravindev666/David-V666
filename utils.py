"""
DAVID PROPHETIC ORACLE — Shared Utilities
==========================================
Constants, formatters, and color helpers for the CLI.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY PATCH (sklearn 1.6+ vs LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════
import sklearn
import sklearn.utils.validation
from packaging import version

_original_check_X_y = sklearn.utils.validation.check_X_y
_original_check_array = sklearn.utils.validation.check_array

# Only patch if sklearn version is 1.6 or higher
if version.parse(sklearn.__version__) >= version.parse("1.6.0"):
    def _patched_check_X_y(X, y, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _original_check_X_y(X, y, **kwargs)

    def _patched_check_array(array, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _original_check_array(array, **kwargs)

    sklearn.utils.validation.check_X_y = _patched_check_X_y
    sklearn.utils.validation.check_array = _patched_check_array
else:
    # On older versions, LightGBM might need the reverse or no patch at all.
    # We stay with original unless we see the "force_all_finite" error later.
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"
SP500_SYMBOL = "^GSPC"
DATA_START_YEAR = 2015
LOT_SIZE = 65  # Nifty lot size (Updated to 65 as per latest NSE standards)

# Direction labels
UP = "UP"
DOWN = "DOWN"
SIDEWAYS = "SIDEWAYS"

# v6.6.6 Mode
BINARY_MODE = True

# Regime labels (5-state)
REGIME_LABELS = {
    0: "STRONG BULLISH",
    1: "MILD BULLISH",
    2: "SIDEWAYS",
    3: "MILD BEARISH",
    4: "STRONG BEARISH",
}

# Direction thresholds for classification
# UP > +0.5%, DOWN < -0.5%, SIDEWAYS in between
DIRECTION_THRESHOLD = 0.005  # ±0.5% — wider zone for SIDEWAYS (was 0.3%)
SIDEWAYS_CONFIDENCE_THRESHOLD = 0.40  # Lowered for credit spreads (they win on sideways too)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI COLORS (ANSI)
# ═══════════════════════════════════════════════════════════════════════════════
class C:
    """ANSI color codes for rich terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"

    @staticmethod
    def up(text):
        return f"{C.GREEN}{C.BOLD}{text}{C.RESET}"
    
    @staticmethod
    def down(text):
        return f"{C.RED}{C.BOLD}{text}{C.RESET}"
    
    @staticmethod
    def neutral(text):
        return f"{C.YELLOW}{C.BOLD}{text}{C.RESET}"
    
    @staticmethod
    def highlight(text):
        return f"{C.CYAN}{C.BOLD}{text}{C.RESET}"
    
    @staticmethod
    def dim(text):
        return f"{C.DIM}{text}{C.RESET}"
    
    @staticmethod
    def header(text):
        return f"{C.MAGENTA}{C.BOLD}{text}{C.RESET}"
    
    @staticmethod  
    def direction_color(direction):
        if direction == UP:
            return C.up(direction)
        elif direction == DOWN:
            return C.down(direction)
        return C.neutral(direction)

    @staticmethod
    def pct(value):
        """Format a probability/percentage with color."""
        s = f"{value:.1f}%"
        if value >= 60:
            return C.up(s)
        elif value <= 40:
            return C.down(s)
        return C.neutral(s)


def banner():
    """Print the David Oracle banner."""
    print(f"""
{C.MAGENTA}{C.BOLD}╔══════════════════════════════════════════════════════════════════╗
║              DAVID PROPHETIC ORACLE v1.0                         ║
║         Nifty Absolute Direction Prediction Engine               ║
║    XGBoost + LightGBM + CatBoost + HMM Ensemble                 ║
╚══════════════════════════════════════════════════════════════════╝{C.RESET}
""")


def separator(title=""):
    """Print a section separator."""
    if title:
        print(f"\n{C.CYAN}{'─'*20} {title} {'─'*20}{C.RESET}")
    else:
        print(f"{C.DIM}{'─'*60}{C.RESET}")


def format_inr(value):
    """Format a number as Indian Rupees."""
    if abs(value) >= 10_000_000:
        return f"₹{value/10_000_000:.2f} Cr"
    elif abs(value) >= 100_000:
        return f"₹{value/100_000:.2f} L"
    return f"₹{value:,.0f}"
