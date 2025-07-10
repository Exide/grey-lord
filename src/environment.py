import ansi
import collections
import gymnasium
import logging
import majormud
import numpy
import queue
import re
import socket
import telnet
import threading
import time
import utils

logger = logging.getLogger(__name__)

SOCKET_READ_TIMEOUT_SECONDS = 0.1 # 100ms
SOCKET_CLOSE_TIMEOUT_SECONDS = 1.0 # 1s
SOCKET_BUFFER_SIZE = 1024
STREAM_ENCODING = 'ascii'


class BBSEnvironment(gymnasium.Env):

    # 'ansi': Returns a string representing the environment's state, typically used for text-based environments.
    metadata = {'render_modes': ['ansi']}

    def __init__(self, host, port, username, password, action_map, tokenizer, max_observations=4, observation_window=512):
        super().__init__()

        self.socket = None
        self.socket_lock = threading.Lock()
        self.reader_thread = None
        self.reader_thread_stop_event = threading.Event()
        self.login_complete_event = threading.Event()

        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.last_message_sent = None

        self.render_buffer = queue.Queue()
        self.agent_buffer = queue.Queue()

        self.action_map = action_map
        self.action_space = gymnasium.spaces.Discrete(len(action_map))

        self.tokenizer = tokenizer
        self.max_observations = max_observations
        self.observation_window = observation_window
        self.observations = collections.deque(maxlen=max_observations)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=(self.max_observations, self.observation_window),
            dtype=numpy.int32
        )

        logger.info('Environment initialized')


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._disconnect()
        self.observations.clear()
        self.last_message_sent = None
        self.login_complete_event.clear()
        self._connect()

        self.reader_thread_stop_event.clear()
        self.reader_thread = threading.Thread(target=self._handle_reader_thread, daemon=True, args=(self.reader_thread_stop_event,))
        self.reader_thread.start()

        # todo: fix timeout magic number
        # wait for login to complete before returning
        if not self.login_complete_event.wait(timeout=30.0):  # 30s
            raise TimeoutError('Login process timed out')

        logger.info('Environment has been reset')
        return self._get_state(), {}


    def step(self, action):
        """Returns the result of the action for the agent to consume."""
        def step_response(step_reward, terminated=False, truncated=False):
            return self._get_state(), step_reward, terminated, truncated, {}

        is_reader_thread_healthy = self.reader_thread and self.reader_thread.is_alive()
        if not is_reader_thread_healthy:
            logger.warning('Reader thread is not healthy, attempting recovery')
            
            try:
                success = self._recover_connection()
                if not success:
                    logger.error('Failed to recover connection')
                    return step_response(-100.0, terminated=True)
                logger.info('Connection recovered successfully')
            except Exception as e:
                logger.exception('Error during connection recovery', exc_info=e)
                return step_response(-100.0, terminated=True)

        if not self.login_complete_event.is_set():
            logger.warning('Cannot step, login not complete')
            return step_response(-1.0)

        command = self.action_map.get(action, None)
        if command is None:
            logger.warning(f'Invalid action ID: {action}')
            return step_response(-1.0)

        try:
            msg = f'{command}\r\n'.encode(STREAM_ENCODING)
            self._send_message(msg)
        except Exception as e:
            logger.exception(f'Failed to send message to the server', exc_info=e)
            return step_response(-100.0, terminated=True)

        # todo: sleep magic number
        time.sleep(0.1) # 100ms
        
        data: bytes = self._get_queued_data()
        if not data: return step_response(0.0) # neutral reward

        observation = self._to_observation(data)
        self.observations.append(observation)
            
        reward = self._calculate_reward(data)
        
        logger.debug(f'Reward: {reward}')

        return step_response(reward)


    def render(self, mode='ansi'):
        """Renders the latest data during training for humans to consume. No impact on training."""
        if mode != 'ansi':
            raise NotImplementedError('Only ANSI mode is supported')

        data: bytes = self._drain_buffer(self.render_buffer)
        if not data: return

        border = '-' * 80
        print(border)
        print(data.decode(STREAM_ENCODING, errors='ignore'))
        print(border)


    def close(self):
        # tell the reader thread to stop
        self.reader_thread_stop_event.set()

        # wait for the thread to stop
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)

        # cleanup the socket
        self._disconnect()


    def _connect(self):
        if self._is_connected(): return

        logger.info(f'Connecting to {self.host}:{self.port}')
        try:
            with self.socket_lock:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(SOCKET_READ_TIMEOUT_SECONDS)

            logger.info('Connection established')

        except Exception as e:
            logger.exception(f'Connection failed', exc_info=e)
            raise


    def _is_connected(self):
        return self.socket is not None


    def _disconnect(self):
        if not self._is_connected(): return

        with self.socket_lock:
            self.socket.close()
            self.socket = None

        logger.info('Connection closed')


    def _recover_connection(self):
        logger.info('Starting connection recovery...')
        
        # get rid of the current reader thread
        if self.reader_thread and self.reader_thread.is_alive():
            logger.info('Stopping current reader thread...')
            self.reader_thread_stop_event.set()
            self.reader_thread.join(timeout=2.0)
            if self.reader_thread.is_alive():
                logger.warning('Reader thread did not stop within timeout')
        
        # reset everything
        self._disconnect()
        self._drain_buffer(self.render_buffer)
        self._drain_buffer(self.agent_buffer)
        self.last_message_sent = None
        self.login_complete_event.clear()
        
        try:
            self._connect()
        except Exception as e:
            logger.exception('Failed to reconnect during recovery', exc_info=e)
            return False
        
        # start a new reader thread
        self.reader_thread_stop_event.clear()
        self.reader_thread = threading.Thread(target=self._handle_reader_thread, daemon=True, args=(self.reader_thread_stop_event,))
        self.reader_thread.start()
        
        # wait for login to complete
        if not self.login_complete_event.wait(timeout=30.0):  # 30s
            logger.error('Login timed out during recovery')
            return False
        
        logger.info('Connection recovery completed successfully')
        return True


    def _handle_reader_thread(self, stop_event: threading.Event):
        is_logged_in = False
        login_timeout_counter = 0
        max_login_timeout = 300  # read_timeout_seconds / login_timeout_seconds (0.1 / 30 = 300)

        telnet_parser = telnet.TelnetParser(self.socket, self.socket_lock)
        ansi_parser = ansi.AnsiParser(self.socket, self.socket_lock)

        while not stop_event.is_set() and self._is_connected():
            try:
                with self.socket_lock:
                    data = self.socket.recv(SOCKET_BUFFER_SIZE)
                if not data:
                    logger.info('Connection closed by the server')
                    return

                data = ansi_parser.handle_ansi_verification(data)

                self.render_buffer.put_nowait(data)
                self.render()

                data = telnet_parser.parse(data)
                data = ansi_parser.tokenize(data)
                
                if not is_logged_in:
                    is_logged_in = self._login(data)
                    if is_logged_in:
                        self.login_complete_event.set()
                    else:
                        login_timeout_counter = 0   # reset the counter
                else:
                    data = self._remove_echo(data)
                    data = self._fix_prompt(data)
                    self.agent_buffer.put_nowait(data)

                logger.debug(f'Data received ({len(data)} bytes): {utils.to_byte_string(data)}')

            except socket.timeout:
                # nothing to read
                if not is_logged_in:
                    login_timeout_counter += 1
                    if login_timeout_counter >= max_login_timeout:
                        logger.error('Login timed out')
                        return
                continue

            except Exception as e:
                logger.exception(f'Error while reading from the socket', exc_info=e)
                return


    def _remove_echo(self, data: bytes) -> bytes:
        def remove(command: bytes, byte_stream: bytes) -> bytes:
            position = byte_stream.find(command)
            byte_stream = byte_stream.replace(command, b'', 1)
            logger.debug(f'Removed command echo ({len(command)} bytes @ offset {position}): {utils.to_byte_string(command)}')
            return byte_stream

        # no command sent
        if not self.last_message_sent:
            return data

        # Converts "command\r\n" to "*******\r\r\n"
        # Notice the quirk of an extra \r in the result.
        # This was added based on what I was seeing in responses from the server.
        last_message_sent_masked: bytes = b'*' * len(self.last_message_sent.strip()) + b'\r\r\n'

        # try command echo removal
        if self.last_message_sent in data:
            output = remove(self.last_message_sent, data)

        # try masked command echo removal
        elif last_message_sent_masked in data:
            output = remove(last_message_sent_masked, data)

        # no echo found
        else:
            output = data

        logger.debug(f'Remaining data is {len(output)} bytes')
        return output


    @staticmethod
    def _fix_prompt(data: bytes) -> bytes:
        """Adds a newline to the end of each prompt received to make it more clear that they're separate info."""
        def add_newline(match) -> bytes:
            full_match = match.group(0)
            return f'{full_match}\r\n'.encode(STREAM_ENCODING)

        return majormud.PROMPT_PATTERN.sub(add_newline, data)


    def _get_queued_data(self) -> bytes:
        data: bytes = self._drain_buffer(self.agent_buffer)

        return data


    def _send_message(self, message: bytes):
        if not self._is_connected(): raise ConnectionError('Cannot send message, connection closed')

        with self.socket_lock:
            self.socket.send(message)
        logger.debug(f'Data sent ({len(message)}): {utils.to_byte_string(message)}')
        logger.info(f'Command sent: {message}')
        self.last_message_sent = message


    def _login(self, data: bytes) -> bool:
        # if majormud.PROMPT_PATTERN.search(data):
        if b'[HP=' in data:
            logger.info('Login complete')
            return True
        
        prompts = [
            (b'already have a User-ID', self.username),
            (b'Enter your password:', self.password),
            (b'(N)onstop, (Q)uit, or (C)ontinue?', 'q'),
            (b'Main System Menu', 'm'),
            (b'Enter the Realm', 'e')
        ]

        try:
            for prompt, command in prompts:
                if prompt in data.decode(STREAM_ENCODING, errors='ignore').encode(STREAM_ENCODING):
                    self._send_message(f'{command}\r\n'.encode(STREAM_ENCODING))
                    break
        except ConnectionError:
            # do nothing, we return False by default
            pass
        
        return False


    def _to_observation(self, data: bytes):
        token_ids = self.tokenizer.encode(data)
        padded_tokens = token_ids[:self.observation_window]
        padding_length = self.observation_window - len(padded_tokens)
        padded_tokens.extend([self.tokenizer.pad_token_id] * padding_length)
        return numpy.array(padded_tokens, dtype=numpy.int32)


    def _get_state(self):
        while len(self.observations) < self.max_observations:
            empty = numpy.full(
                shape=self.observation_window,
                fill_value=self.tokenizer.pad_token_id,
                dtype=numpy.int32
            )
            self.observations.appendleft(empty)
        return numpy.array(list(self.observations), dtype=numpy.int32)


    def _calculate_reward(self, data: bytes):
        reward = 0.0

        if not data: return reward

        # positive rewards
        reward += majormud.reward_for_reaching_level(data)
        reward += majormud.reward_for_experience(data)
        reward += majormud.reward_for_damage_done(data)
        reward += majormud.reward_for_dodging_attacks(data)
        reward += majormud.reward_for_avoiding_attacks(data)

        # negative rewards
        reward -= majormud.reward_for_taking_damage(data)
        reward -= majormud.reward_for_dropping(data)
        reward -= majormud.reward_for_dying(data)
        reward -= majormud.reward_for_invalid_command(data)

        return reward





    @staticmethod
    def _drain_buffer(buffer: queue.Queue) -> bytes:
        output = b''
        while not buffer.empty():
            try:
                output += buffer.get_nowait()
            except queue.Empty:
                break
        return output
