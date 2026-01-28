from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

def parse_winning_numbers_field(s: str) -> Tuple[List[int], Optional[int]]:
    """
    Socrata datasets store winning_numbers like:
      '01 04 13 21 35' (5 main numbers, bonus in separate field)
    or '01 04 13 21 35 12' (5 main + bonus together)
    We'll parse into ([main...], bonus or None)
    """
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    nums = [int(p) for p in parts]
    if len(nums) < 5:
        raise ValueError(f"Unexpected winning_numbers format: {s}")
    main = nums[:5]
    bonus = nums[5] if len(nums) >= 6 else None
    return main, bonus