import re
from typing import Optional, Tuple


def infer_params_from_name(name_or_id: str) -> Optional[float]:
    """
    Extracts model parameter count in billions (B) from model IDs/names like
    'Llama-3.2-3B-Instruct', 'Qwen2.5-72B', 'Mixtral-8x7B' -> 46.7B, 'DeepSeek-R1-Distill-Qwen-1.5B'.
    """
    moe_match = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]", name_or_id)
    if moe_match:
        experts = float(moe_match.group(1))
        per_expert = float(moe_match.group(2))
        return round(experts * per_expert * 0.85, 1)

    param_match = re.search(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)\s*[bB](?:[-_/ ]|$)", name_or_id)
    if param_match:
        val = float(param_match.group(1))
        if 0.1 <= val <= 2000.0:
            return val
    return None


def calculate_vram_footprints(
    parameters_b: Optional[float], context_length: int = 8192
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculates realistic GPU VRAM requirements (in GB) including KV cache overhead:
    Returns (vram_q4_gb, vram_q8_gb, vram_fp16_gb).
    """
    if not parameters_b or parameters_b <= 0:
        return None, None, None

    kv_overhead = min(2.5, max(0.5, (context_length / 32768.0) * 0.6))
    vram_q4 = round(parameters_b * 0.62 + kv_overhead, 1)
    vram_q8 = round(parameters_b * 1.12 + kv_overhead * 1.2, 1)
    vram_fp16 = round(parameters_b * 2.15 + kv_overhead * 1.5, 1)
    return vram_q4, vram_q8, vram_fp16
