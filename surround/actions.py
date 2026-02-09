ACTION_WORDS_4 = ("UP", "RIGHT", "LEFT", "DOWN")
ACTION_WORD_TO_ID = {word: action_id + 1 for action_id, word in enumerate(ACTION_WORDS_4)}
STATES = (
    "D_UP",
    "D_RIGHT",
    "D_LEFT",
    "D_DOWN",
    "REL_X_OPP",
    "REL_Y_OPP",
)
