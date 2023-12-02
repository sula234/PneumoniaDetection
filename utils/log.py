from colorlog import ColoredFormatter
import logging


# Create a logger
logger = logging.getLogger("example")
logger.setLevel(logging.DEBUG)

# Define a custom formatter with time and square brackets
formatter = ColoredFormatter(
    "[%(cyan)s%(asctime)s%(reset)s]%(log_color)s[%(levelname)s]%(reset)s%(cyan)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)