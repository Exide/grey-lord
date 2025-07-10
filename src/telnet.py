import logging
import utils


logger = logging.getLogger(__name__)

# Telnet command constants
IAC  = b'\xff' # 255
DONT = b'\xfe' # 254
DO   = b'\xfd' # 253
WONT = b'\xfc' # 252
WILL = b'\xfb' # 251

# Telnet option constants
BINARY = b'\x00' # 0
ECHO   = b'\x01' # 1
SGA    = b'\x03' # 3 (Suppress Go Ahead)

POTENTIAL_TOKENS = [
    '<|TELNET#UNKNOWN|>',
    '<|TELNET#WILL_BINARY|>',
    '<|TELNET#WONT_BINARY|>',
    '<|TELNET#DO_BINARY|>',
    '<|TELNET#DONT_BINARY|>',
    '<|TELNET#WILL_ECHO|>',
    '<|TELNET#WONT_ECHO|>',
    '<|TELNET#DO_ECHO|>',
    '<|TELNET#DONT_ECHO|>',
    '<|TELNET#WILL_SGA|>',
    '<|TELNET#WONT_SGA|>',
    '<|TELNET#DO_SGA|>',
    '<|TELNET#DONT_SGA|>'
]


class TelnetParser:

    def __init__(self, sock, lock):
        self.socket = sock
        self.socket_lock = lock

    def update_socket(self, sock, lock):
        self.socket = sock
        self.socket_lock = lock

    def parse(self, data: bytes) -> bytes:
        iac_index = data.find(IAC)
        if iac_index < 0: return data

        last_index = len(data) - 1
        is_last_character = iac_index == last_index
        if is_last_character: return data

        has_room = iac_index + 2 <= last_index
        if not has_room: return data

        command_index = iac_index + 1
        option_index = iac_index + 2
        after_index = option_index + 1 if option_index <= len(data) else option_index

        command: bytes = data[command_index:option_index]
        option: bytes = data[option_index:after_index]
        msg: bytes = IAC + command + option

        logger.debug(f'Telnet command received ({len(msg)} bytes): {utils.to_byte_string(msg)}')

        if command == WILL:
            if option in [BINARY, SGA]:
                self.send_command(DO, option)
            else:
                self.send_command(DONT, option)

        if command == DO:
            if option in [BINARY, SGA]:
                self.send_command(WILL, option)
            else:
                self.send_command(WONT, option)

        # replace the telnet sequence with its token
        token = b'<|TELNET#' + command + option + b'|>'
        before = data[0:iac_index]
        after = data[option_index:]
        data = before + token + after

        return data

    def send_command(self, command: bytes, option: bytes):
        msg = IAC + command + option
        with self.socket_lock:
            self.socket.send(msg)
        logger.debug(f'Telnet command sent ({len(msg)} bytes): {utils.to_byte_string(msg)}')
