import logging
from typing import Dict, List, Union, Optional, Tuple
from transformers import GPT2Tokenizer
from ansi import POTENTIAL_TOKENS as ANSI_TOKENS
from telnet import POTENTIAL_TOKENS as TELNET_TOKENS

logger = logging.getLogger(__name__)

class GreyLordTokenizer:
    """
    Hybrid tokenizer combining GPT-2's English understanding with special token handling.
    
    Token ID ranges:
    - GPT-2 base tokens: 0-50256 (standard GPT-2 vocabulary)
    - Special game tokens: 50257-50357 (ANSI/Telnet/PAD tokens)
    - Byte fallback tokens: 50358-50613 (256 byte tokens for unknown sequences)
    
    This gives the model built-in English understanding while preserving
    the ability to handle raw game protocol sequences.
    """
    
    def __init__(self):
        # Initialize base GPT-2 tokenizer
        self.base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Token mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Special token markers for preprocessing
        self.special_token_markers: Dict[str, str] = {}
        
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build the hybrid vocabulary with organized token ID ranges"""
        
        # Range 0-50256: GPT-2 base tokens
        gpt2_vocab = self.base_tokenizer.get_vocab()
        self.token_to_id.update(gpt2_vocab)
        self.id_to_token.update({v: k for k, v in gpt2_vocab.items()})
        
        # Range 50257+: Special game tokens
        special_start_id = 50257
        current_id = special_start_id
        
        # Add ANSI tokens
        ansi_start_id = current_id
        for token in ANSI_TOKENS:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            # Create unique marker for preprocessing
            self.special_token_markers[token] = f"__SPECIAL_TOKEN_{current_id}__"
            current_id += 1
        ansi_stop_id = current_id - 1
        
        # Add Telnet tokens
        telnet_start_id = current_id
        for token in TELNET_TOKENS:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            # Create unique marker for preprocessing
            self.special_token_markers[token] = f"__SPECIAL_TOKEN_{current_id}__"
            current_id += 1
        telnet_stop_id = current_id - 1
        
        # Add control tokens
        control_start_id = current_id
        control_tokens = ['<|PAD|>']
        for token in control_tokens:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            # PAD doesn't need a marker as it's not in raw data
            current_id += 1
        control_stop_id = current_id - 1
        
        # Range 50358-50613: Byte fallback tokens
        byte_start_id = current_id
        for i in range(256):
            token = f'<|BYTE_{i}|>'
            self.token_to_id[token] = byte_start_id + i
            self.id_to_token[byte_start_id + i] = token
        byte_stop_id = byte_start_id + 255
        
        # Calculate vocab size
        self._vocab_size = max(self.id_to_token.keys()) + 1
        
        logger.info(f'GreyLord tokenizer initialized with {self._vocab_size} tokens')
        logger.info('Token ranges: ' + ', '.join([
            f'GPT-2(0-{max(gpt2_vocab.values())})',
            f'ANSI({ansi_start_id}-{ansi_stop_id})',
            f'Telnet({telnet_start_id}-{telnet_stop_id})',
            f'Control({control_start_id}-{control_stop_id})',
            f'Bytes({byte_start_id}-{byte_stop_id})'
        ]))

    def _preprocess_special_tokens(self, data: Union[bytes, str]) -> Tuple[str, Dict[str, int]]:
        """
        Replace special tokens with unique markers and track their positions.
        
        Returns:
            Tuple of (processed_text, marker_to_token_id_map)
        """
        if isinstance(data, bytes):
            # Try to decode as UTF-8, fall back to latin-1 for robustness
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                text = data.decode('latin-1')
        else:
            text = data
            
        marker_map = {}
        
        # Replace special tokens with markers (longest first to avoid conflicts)
        special_tokens_sorted = sorted(self.special_token_markers.keys(), key=len, reverse=True)
        
        for token in special_tokens_sorted:
            if token in text:
                marker = self.special_token_markers[token]
                text = text.replace(token, marker)
                marker_map[marker] = self.token_to_id[token]
        
        return text, marker_map

    def _handle_unknown_bytes(self, text: str) -> List[int]:
        """
        Handle any remaining unknown characters by converting to byte tokens.
        """
        token_ids = []
        
        for char in text:
            try:
                # Try to encode the character
                char_bytes = char.encode('utf-8')
                for byte_val in char_bytes:
                    byte_token = f'<|BYTE_{byte_val}|>'
                    token_ids.append(self.token_to_id[byte_token])
            except:
                # If all else fails, use a null byte token
                token_ids.append(self.token_to_id['<|BYTE_0|>'])
                
        return token_ids

    def encode(self, data: Union[bytes, str]) -> List[int]:
        """
        Encode data using hybrid approach:
        1. Identify special tokens in the text
        2. Split text into segments around special tokens
        3. Use GPT-2 tokenizer on regular text segments
        4. Insert special token IDs at appropriate positions
        """
        if isinstance(data, bytes):
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                text = data.decode('latin-1')
        else:
            text = data
            
        # Find all special tokens in the text
        special_tokens_in_text = []
        for token in sorted(self.special_token_markers.keys(), key=len, reverse=True):
            start = 0
            while True:
                pos = text.find(token, start)
                if pos == -1:
                    break
                special_tokens_in_text.append((pos, pos + len(token), token))
                start = pos + len(token)
        
        # Sort by position
        special_tokens_in_text.sort()
        
        # If no special tokens, just use GPT-2 tokenizer
        if not special_tokens_in_text:
            try:
                return self.base_tokenizer.encode(text, add_special_tokens=False)
            except Exception as e:
                logger.warning(f"GPT-2 tokenization failed: {e}, falling back to byte tokens")
                return self._handle_unknown_bytes(text)
        
        # Split text around special tokens and tokenize
        final_token_ids = []
        last_end = 0
        
        for start, end, token in special_tokens_in_text:
            # Tokenize text before this special token
            if start > last_end:
                text_segment = text[last_end:start]
                if text_segment:
                    try:
                        tokens = self.base_tokenizer.encode(text_segment, add_special_tokens=False)
                        final_token_ids.extend(tokens)
                    except Exception as e:
                        logger.warning(f"GPT-2 tokenization failed on segment: {e}")
                        final_token_ids.extend(self._handle_unknown_bytes(text_segment))
            
            # Add the special token ID
            special_token_id = self.token_to_id[token]
            final_token_ids.append(special_token_id)
            last_end = end
        
        # Tokenize remaining text after the last special token
        if last_end < len(text):
            text_segment = text[last_end:]
            if text_segment:
                try:
                    tokens = self.base_tokenizer.encode(text_segment, add_special_tokens=False)
                    final_token_ids.extend(tokens)
                except Exception as e:
                    logger.warning(f"GPT-2 tokenization failed on final segment: {e}")
                    final_token_ids.extend(self._handle_unknown_bytes(text_segment))
        
        return final_token_ids

    def decode(self, token_ids: List[int]) -> bytes:
        """
        Decode token IDs back to bytes.
        """
        result_parts = []
        
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                logger.warning(f'Unknown token ID: {token_id}')
                continue
                
            token = self.id_to_token[token_id]
            
            # Handle special control tokens
            if token == '<|PAD|>':
                continue  # Skip padding tokens
                
            # Handle byte tokens
            elif token.startswith('<|BYTE_') and token.endswith('|>'):
                byte_value = int(token[7:-2])
                result_parts.append(bytes([byte_value]))
                
            # Handle GPT-2 tokens
            elif token_id <= 50256:
                try:
                    decoded_text = self.base_tokenizer.decode([token_id])
                    result_parts.append(decoded_text.encode('utf-8'))
                except:
                    # If GPT-2 decode fails, treat as unknown
                    result_parts.append(b'?')
                    
            # Handle special game tokens (ANSI/Telnet)
            else:
                result_parts.append(token.encode('utf-8'))
                
        return b''.join(result_parts)

    def get_vocab_size(self) -> int:
        """Get the total vocabulary size"""
        return self._vocab_size
        
    def get_token_id(self, token: str) -> Optional[int]:
        """Get token ID for a specific token"""
        return self.token_to_id.get(token)
        
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string for a specific token ID"""
        return self.id_to_token.get(token_id)
        
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get mapping of special token names to IDs"""
        return {
            'pad': self.token_to_id['<|PAD|>'],
        }
        
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID (compatibility with transformers API)"""
        return self.token_to_id['<|PAD|>']
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (compatibility with transformers API)"""
        return self._vocab_size
        
    def get_byte_token_id(self, byte_value: int) -> int:
        """Get token ID for a specific byte value (0-255)"""
        if not 0 <= byte_value <= 255:
            raise ValueError(f"Byte value must be 0-255, got {byte_value}")
        return 50358 + byte_value  # Byte tokens start at 50358
        
    def get_token_ranges(self) -> Dict[str, tuple]:
        """Get the token ID ranges for different token types"""
        gpt2_max = max(k for k in self.id_to_token.keys() if k <= 50256)
        return {
            'gpt2': (0, gpt2_max),
            'ansi': (50257, 50257 + len(ANSI_TOKENS) - 1),
            'telnet': (50257 + len(ANSI_TOKENS), 50257 + len(ANSI_TOKENS) + len(TELNET_TOKENS) - 1),
            'control': (50257 + len(ANSI_TOKENS) + len(TELNET_TOKENS), 50257 + len(ANSI_TOKENS) + len(TELNET_TOKENS)),
            'bytes': (50358, 50613)
        }
        
    def batch_encode(self, data_list: List[Union[bytes, str]], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Encode multiple sequences with optional padding
        
        Args:
            data_list: List of bytes or strings to encode
            max_length: Optional maximum length for padding/truncation
            
        Returns:
            List of token ID lists
        """
        encoded_sequences = []
        
        for data in data_list:
            token_ids = self.encode(data)
            
            if max_length:
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                else:
                    pad_id = self.token_to_id['<|PAD|>']
                    token_ids.extend([pad_id] * (max_length - len(token_ids)))
                    
            encoded_sequences.append(token_ids)
            
        return encoded_sequences
        
    def batch_decode(self, token_ids_list: List[List[int]]) -> List[bytes]:
        """
        Decode multiple sequences of token IDs
        
        Args:
            token_ids_list: List of token ID lists to decode
            
        Returns:
            List of decoded bytes
        """
        return [self.decode(token_ids) for token_ids in token_ids_list]

    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        import json
        
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self._vocab_size,
            'token_ranges': self.get_token_ranges()
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
            
        logger.info(f'Vocabulary saved to {filepath}')
        
    @classmethod
    def load_vocab(cls, filepath: str) -> 'GreyLordTokenizer':
        """Load vocabulary from file"""
        import json
        
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
            
        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = vocab_data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in vocab_data['token_to_id'].items()}
        tokenizer._vocab_size = vocab_data['vocab_size']
        
        logger.info(f'Vocabulary loaded from {filepath}')
        return tokenizer

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value
