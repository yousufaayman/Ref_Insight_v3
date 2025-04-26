EVENT_DICTIONARY = {
    "action_class": {
        "Standing tackling": 0,
        "Tackling": 1,
        "High leg": 2,
        "Pushing": 3,
        "Holding": 4,
        "Elbowing": 5,
        "Challenge": 6,
        "Dive/Simulation": 7
    },
    "offense_severity": {
        "No offence": 0,
        "Offence + No card": 1,
        "Offence + Yellow card": 2,
        "Offence + Red card": 3
    }
}

SEVERITY_MAPPING = {
    "1.0": "Offence + No card",
    "3.0": "Offence + Yellow card",
    "5.0": "Offence + Red card"
}

NUM_FOUL_TYPES = 8
NUM_OFFENSE_CATEGORIES = 4