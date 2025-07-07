def to_byte_string(buffer: bytes) -> str:
    return ' '.join(f'{b:02x}' for b in buffer)
