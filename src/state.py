import threading


class SharedWorldState:

    def __init__(self):
        self.lock = threading.Lock()
        self.log = b''
        self.player_hitpoints = 0
        self.player_mana = 0
        self.player_is_resting = False
        self.player_is_meditating = False


    def update(self, parsed_data):
        with self.lock:
            if parsed_data.player_hitpoints:
                self.player_hitpoints = parsed_data.player_hitpoints
            if parsed_data.player_mana:
                self.player_mana = parsed_data.player_mana
            if parsed_data.player_is_resting:
                self.player_is_resting = parsed_data.player_is_resting
            if parsed_data.player_is_meditating:
                self.player_is_meditating = parsed_data.player_is_meditating


    def get_observation(self):
        """Returns a copy of the data so the lock can be released."""
        with self.lock:
            return {
                'player_hitpoints': self.player_hitpoints,
                'player_mana': self.player_mana,
                'player_is_resting': self.player_is_resting,
                'player_is_meditating': self.player_is_meditating
            }
