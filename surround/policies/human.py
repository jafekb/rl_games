from surround.actions import ACTION_WORD_TO_ID


def get_human_action(action_space, observation, info) -> int:
    return ACTION_WORD_TO_ID["RIGHT"]
