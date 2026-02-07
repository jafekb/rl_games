from surround.actions import ACTION_WORD_TO_ID


def get_ai_action(action_space, action_names=None, observation=None, info=None):
    directional = [ACTION_WORD_TO_ID[name] for name in ("UP", "RIGHT", "LEFT", "DOWN")]
    return directional[action_space.sample() % len(directional)]
