ACTIONS_BY_ID = {
    0: 'help',
    # Navigation
    1: 'north',
    2: 'northeast', 
    3: 'east',
    4: 'southeast',
    5: 'south',
    6: 'southwest',
    7: 'west',
    8: 'northwest',
    9: 'up',
    10: 'down',
    # Metadata
    100: 'stat',
    101: 'inventory',
    102: 'experience',
    103: 'spells',
    # Interaction
    200: '',
    201: 'look',
    202: 'attack',
    203: 'bash',
    204: 'sneak',
    205: 'hide',
    206: 'get',
    207: 'drop',
    208: 'use',
    209: 'equip',
    210: 'unequip',
    211: 'read',
    # Combat
    # This is to simplify combat for the initial training.
    # This will be removed once we support compound actions.
    250: 'attack kobold thief',
    251: 'attack giant rat',
    252: 'attack acid slime',
    253: 'attack orc rogue',
    254: 'attack cave bear'
}

# Reverse mapping for easy lookup
ACTIONS_BY_COMMAND = {v: k for k, v in ACTIONS_BY_ID.items()}
