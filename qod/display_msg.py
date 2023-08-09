reset_color = "\033[0m"
red = "\033[0;31m"
green = "\033[0;32m"
yellow = "\033[0;33m"
blue = "\033[0;34m"
purple = "\033[0;35m"
cyan = "\033[0;36m"
white = "\033[0;37m"
b_red = "\033[1;31m"
b_green = "\033[1;32m"
b_yellow = "\033[1;33m"
b_blue = "\033[1;34m"
b_purple = "\033[1;35m"
b_cyan = "\033[1;36m"
b_white = "\033[1;37m"

COLOR_ERROR = b_red
COLOR_CLI_QUERY = cyan
COLOR_CLI_NOTIFICATION = white
COLOR_LLM_NOTIFICATION = purple
COLOR_CHAIN_TEMP = yellow
COLOR_CHAIN_FINAL = blue


def display_error(msg: str, reset: bool = True) -> None:
    """Display a message in the color for an error
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_ERROR}{msg}"
    if reset:
        text += f"{reset_color}"
    print(text)


def cli_input(msg: str, reset: bool = True) -> str:
    """Display a message in the color for a CLI query
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_CLI_QUERY}{msg}"
    if reset:
        text += f"{reset_color}"
    value = input(text)
    return value


def display_cli_notification(msg: str, reset: bool = True) -> None:
    """Display a message in the color for a CLI notification
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_CLI_NOTIFICATION}{msg}"
    if reset:
        text += f"{reset_color}"
    print(text)


def display_llm_notification(msg: str, reset: bool = True) -> None:
    """Display a message in the color for a LLM notification
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_LLM_NOTIFICATION}{msg}"
    if reset:
        text += f"{reset_color}"
    print(text)


def display_chain_temp(msg: str, reset: bool = True) -> None:
    """Display a message in the color for a intermediary chain notification
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_CHAIN_TEMP}{msg}"
    if reset:
        text += f"{reset_color}"
    print(text)


def display_chain_final(msg: str, reset: bool = True) -> None:
    """Display a message in the color for a intermediary chain notification
    :param msg: Message to display
    :param reset: Boolean indicating if the color must be\
    reset to the default color once the message is displayed
    """
    text = f"{COLOR_CHAIN_FINAL}{msg}"
    if reset:
        text += f"{reset_color}"
    print(text)
