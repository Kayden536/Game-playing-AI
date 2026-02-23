import logging
import datetime

def setup_logger(name="GameAI"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(
        f"logs/game_ai_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger