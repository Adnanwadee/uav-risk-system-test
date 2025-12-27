# src/uav_risk/stage1/utils.py

def to_string_safe(x):
    """
    Must match EXACTLY the function used in Stage-1 preprocessing
    """
    try:
        return x.astype(str)
    except Exception:
        return x
