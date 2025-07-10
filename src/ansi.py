import logging
import re
import utils


logger = logging.getLogger(__name__)

SELECT_GRAPHIC_RENDITION = {
    0: '<|ANSI#RESET|>',
    1: '<|ANSI#BOLD|>',
    4: '<|ANSI#UNDERLINE|>',
    5: '<|ANSI#BLINK|>',
    30: '<|ANSI#FG_BLACK|>',
    31: '<|ANSI#FG_RED|>',
    32: '<|ANSI#FG_GREEN|>',
    33: '<|ANSI#FG_YELLOW|>',
    34: '<|ANSI#FG_BLUE|>',
    35: '<|ANSI#FG_MAGENTA|>',
    36: '<|ANSI#FG_CYAN|>',
    37: '<|ANSI#FG_WHITE|>',
    40: '<|ANSI#BG_BLACK|>',
    41: '<|ANSI#BG_RED|>',
    42: '<|ANSI#BG_GREEN|>',
    43: '<|ANSI#BG_YELLOW|>',
    44: '<|ANSI#BG_BLUE|>',
    45: '<|ANSI#BG_MAGENTA|>',
    46: '<|ANSI#BG_CYAN|>',
    47: '<|ANSI#BG_WHITE|>'
}

# CURSOR_POSITION_REPORT = '<|ANSI#CPR_row_column|>'

DEVICE_STATUS_REPORT: bytes = b'\x1b\x5b\x36\x6e'              # ESC[6n
CURSOR_POSITION_REPORT: bytes = b'\x1b\x5b\x31\x3b\x31\x52'    # ESC[1;1R (row 1, column 1)

# Comprehensive ANSI CSI sequence pattern following ECMA-48 standard
# \x1b\[ - CSI introducer (ESC [)
# ([\x30-\x3F]*?) - Parameter bytes (0-9, :, ;, <, =, >, ?) - zero or more
# ([\x20-\x2F]*?) - Intermediate bytes (space through /) - zero or more  
# ([\x40-\x7E]) - Final byte (@ through ~) - exactly one
# This catches: SGR, CPR, DSR, cursor movement, private sequences, etc.
REGEX_PATTERN = re.compile(rb'\x1b\[([\x30-\x3F]*?)([\x20-\x2F]*?)([\x40-\x7E])')


class AnsiParser:

    def __init__(self, sock, lock):
        self.socket = sock
        self.socket_lock = lock


    def update_socket(self, sock, lock):
        self.socket = sock
        self.socket_lock = lock


    def tokenize(self, data: bytes) -> bytes:
        def create_token(path) -> bytes:
            full_match = path.group(0)      # The entire match including \x1b[ and ending char
            parameters = path.group(1)      # Parameters part (e.g., "31" or "1;32" or "?25")
            intermediates = path.group(2)   # Intermediate bytes (space through /)
            ending_char = path.group(3)     # The ending character ("m" or "R")

            logger.debug(f'ANSI sequence found ({len(full_match)} bytes): {utils.to_byte_string(full_match)}')

            match ending_char:
                case b'm': return self.tokenize_select_graphic_rendition(full_match, parameters).encode('utf-8')
                case b'R': return self.tokenize_cursor_position_report(full_match).encode('utf-8')
                case _: return self.tokenize_generic(full_match).encode('utf-8')

        return REGEX_PATTERN.sub(create_token, data)


    def handle_ansi_verification(self, data: bytes) -> bytes:
        # check for a DSR request
        dsr_position = data.find(DEVICE_STATUS_REPORT)
        if dsr_position < 0: return data
        logger.debug(f'ANSI Device Status Report received: {utils.to_byte_string(DEVICE_STATUS_REPORT)}')

        # send our CSR response
        with self.socket_lock:
            self.socket.send(CURSOR_POSITION_REPORT)
        logger.debug(f'ANSI Cursor Position Report sent (row 1, col 1): {utils.to_byte_string(CURSOR_POSITION_REPORT)}')

        # remove the sequence from the stream
        return data.replace(DEVICE_STATUS_REPORT, b'')


    @staticmethod
    def tokenize_cursor_position_report(data: bytes) -> str:
        # For now, just return the hex representation like in training data
        return AnsiParser.tokenize_generic(data)


    @staticmethod
    def tokenize_select_graphic_rendition(data: bytes, codes: bytes) -> str:
        if not codes:
            return SELECT_GRAPHIC_RENDITION.get(0, '')

        parts = codes.split(b';')
        tokens = []
        for part in parts:
            if not part.isdigit():
                logger.warning(f'Non-numeric ANSI code: {part}')
                continue

            code = int(part)
            token = SELECT_GRAPHIC_RENDITION.get(code, '')
            if not token:
                logger.warning(f'Unknown ANSI code: {code}')
                continue

            tokens.append(token)

        replacement = ''.join(tokens)
        logger.debug(f'Replacing {repr(data)} with {repr(replacement)}')
        return replacement


    @staticmethod
    def tokenize_generic(data: bytes) -> str:
        value = data.hex(' ', 1).upper()
        return f'<|ANSI#{value}|>'

