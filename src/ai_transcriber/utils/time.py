def format_time(seconds: float, include_ms: bool = False) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if include_ms:
        milliseconds = int((seconds - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"

    return f"{hours:02}:{minutes:02}:{secs:02}"
