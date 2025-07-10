import logging
from typing import Dict, List, Union, Optional
from ansi import POTENTIAL_TOKENS as ANSI_TOKENS
from telnet import POTENTIAL_TOKENS as TELNET_TOKENS


logger = logging.getLogger(__name__)


class GreyLordTokenizer:
    """
    GreyLord tokenizer for processing raw BBS game data with complete byte coverage.
    
    Token ID ranges:
    - Byte tokens (0x00-0xFF) → Token IDs 0-255
    - Telnet tokens → Token IDs 300+
    - ANSI tokens → Token IDs 400+
    - Special tokens → Token ID 500 (PAD only)
    """
    
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build the complete vocabulary with organized token ID ranges"""
        
        # Range 0-255: Byte tokens (BYTE_0 through BYTE_255)
        for i in range(256):
            token = f'<|BYTE_{i}|>'
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # Range 300+: Telnet tokens
        telnet_start_id = 300
        telnet_stop_id = telnet_start_id + len(TELNET_TOKENS) - 1
        for idx, token in enumerate(TELNET_TOKENS):
            token_id = telnet_start_id + idx
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            
        # Range 400+: ANSI tokens
        ansi_start_id = 400
        ansi_stop_id = ansi_start_id + len(ANSI_TOKENS) - 1
        for idx, token in enumerate(ANSI_TOKENS):
            token_id = ansi_start_id + idx
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            
        # Range 500+: Special control tokens
        special_tokens = [
            '<|PAD|>',      # Padding token for batching sequences to same length
        ]
        special_start_id = 500
        special_stop_id = special_start_id + len(special_tokens) - 1
        for idx, token in enumerate(special_tokens):
            token_id = special_start_id + idx
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        # Calculate vocab size based on highest token ID
        self.vocab_size = max(self.id_to_token.keys()) + 1
        
        logger.info(f'GreyLord tokenizer initialized with {self.vocab_size} tokens')
        logger.info('Token ranges: ', ', '.join([
                'Bytes(0-255)',
                f'Telnet({telnet_start_id}-{telnet_stop_id})',
                f'ANSI({ansi_start_id}-{ansi_stop_id})',
                f'Special({special_start_id}-{special_stop_id})'
            ]))

    def encode(self, data: Union[bytes, str]) -> List[int]:
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        token_ids = []
        i = 0
        
        while i < len(data):
            # Try to match special tokens first (ANSI/Telnet)
            matched = False
            
            # Look for the longest possible special token match
            for token_len in range(min(50, len(data) - i), 0, -1):  # Max reasonable token length
                candidate = data[i:i + token_len].decode('utf-8', errors='ignore')
                if candidate in self.token_to_id:
                    token_ids.append(self.token_to_id[candidate])
                    i += token_len
                    matched = True
                    break
                    
            # If no special token matched, encode as byte token
            if not matched:
                byte_value = data[i]
                # Byte tokens have IDs 0-255, matching their byte values
                token_ids.append(byte_value)
                i += 1
                
        return token_ids
        
    def decode(self, token_ids: List[int]) -> bytes:
        result = b''
        
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                logger.warning(f'Unknown token ID: {token_id}')
                continue
                
            token = self.id_to_token[token_id]
            
            # Handle special control tokens
            if token == '<|PAD|>':
                continue  # Skip padding tokens in decoding
                
            # Handle byte tokens (token IDs 0-255 map directly to byte values)
            elif token.startswith('<|BYTE_') and token.endswith('|>'):
                byte_value = int(token[7:-2])  # Extract number from <|BYTE_X|>
                result += bytes([byte_value])
                
            # Handle ANSI/Telnet tokens - convert back to original sequences
            else:
                result += token.encode('utf-8')
                
        return result
        
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size"""
        return self.vocab_size
        
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
        return self.get_vocab_size()
        
    def get_byte_token_id(self, byte_value: int) -> int:
        """Get token ID for a specific byte value (0-255)"""
        if not 0 <= byte_value <= 255:
            raise ValueError(f"Byte value must be 0-255, got {byte_value}")
        return byte_value  # Byte tokens have IDs matching their values
        
    def get_token_ranges(self) -> Dict[str, tuple]:
        """Get the token ID ranges for different token types"""
        return {
            'bytes': (0, 255),
            'telnet': (300, 300 + len(TELNET_TOKENS) - 1),
            'ansi': (400, 400 + len(ANSI_TOKENS) - 1),
            'special': (500, 500),  # 1 special token (PAD)
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
            'vocab_size': self.vocab_size,
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
        tokenizer.vocab_size = vocab_data['vocab_size']
        
        logger.info(f'Vocabulary loaded from {filepath}')
        return tokenizer 