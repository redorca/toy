"Dave's favorite log formats all in one place"
import sys
import pathlib
import logging
import logging.handlers
from textwrap import dedent

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

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
ignore = """
{funcName:s} function name
{name:s} logger name, as given. might be empty
{filename:s} file name including .py
{module:s} module is filename without .py
"""
# del ignore

logformat = logging.Formatter(
    "{levelname:.1s}_{module:s}_{lineno:03d}: {message}",
    style="{",
)


def filehandler(logfilename, level=logging.DEBUG, formatter=logformat):
    file_handler = logging.handlers.TimedRotatingFileHandler(
        str(logfilename), when="midnight", backupCount=30
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    return file_handler


# def name(dunder_file) -> str:
#     # converts __file__ to name of file
#     # print(dunder_file)
#     return pathlib.Path(dunder_file).stem


# print(f"__file__ is {name(__file__)}") # file stem: davelogging
# print(f"__name__ is {name(__name__)}") # package naming: streamer.davelogging


def logger(name, level, logformat=logformat):
    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    logger_.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logformat)
    logger_.addHandler(console_handler)
    return logger_


if __name__ == "__main__":

    from pathlib import Path

    script_name = Path(__file__).stem
    logger = logger(
        script_name,
        logging.DEBUG,
    )

    jhandler = filehandler(
        "/home/davs2rt/downloads/jsonformat.log",
        level=logging.DEBUG,
        formatter=jsonlogformat,
    )
    logger.addHandler(jhandler)

    logger.debug("my debug message")
    logger.warning("my warning message")
    logger.info("my information message")
