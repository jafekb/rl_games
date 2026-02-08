ACTION_WORDS_5 = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
ACTION_WORD_TO_ID = {word: action_id for action_id, word in enumerate(ACTION_WORDS_5)}
STATES = (
    "D_UP",
    "D_DOWN",
    "D_LEFT",
    "D_RIGHT",
    "REL_X_OPP",
    "REL_Y_OPP",
)
