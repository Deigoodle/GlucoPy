FACTOR = 18.0

def mgdl_to_mmoll(mgdl):
    """Convert mg/dL to mmol/L."""

    return mgdl / FACTOR

def mmoll_to_mgdl(mmoll):
    """Convert mmol/L to mg/dL."""

    return mmoll * FACTOR