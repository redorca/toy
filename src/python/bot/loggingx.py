"My favorite log formats all in one place"
import pathlib
import sys
import logging.handlers
from textwrap import dedent

jsonlogformat = logging.Formatter(
    dedent(
        """{{
        "level": "{levelname:s}",
        "file": "{name:s}",
        "func": "{funcName:s}",
        "line": {lineno:d},
        "msg": "{message:s}"
        }},"""
    )
    .replace("\n", "")
    .replace(" ", ""),
    style="{",
)

logformat = logging.Formatter(
    "{levelname:.3s}:{filename:s}:{lineno:03d}: {message}", style="{",
) #{funcName:s}:


def filehandler(logfilename, level=logging.DEBUG, logformat=logformat):
    file_handler = logging.handlers.TimedRotatingFileHandler(
        str(logfilename), when="midnight", backupCount=30
    )
    file_handler.setFormatter(logformat)
    file_handler.setLevel(level=level)
    return file_handler


def _name(dunder_file) -> str:
    # converts __file__ to name of file
    print(dunder_file)
    return pathlib.Path(dunder_file).name


def logger(name, level, logformat=logformat):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logformat)
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    import sys
    from pathlib import Path

    script_name = Path(__file__).stem
    logger = logger(script_name, logging.DEBUG,)

    jhandler = filehandler(
        "/home/davs2rt/downloads/jsonformat.log",
        level=logging.DEBUG,
        logformat=jsonlogformat,
    )
    logger.addHandler(jhandler)

    logger.debug("my debug message")
    logger.warning("my warning message")
    logger.info("my information message")
