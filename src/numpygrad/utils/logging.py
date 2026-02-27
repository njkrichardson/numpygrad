import logging
from pathlib import Path

from numpygrad.configuration import DEFAULT_STREAM_LOG_LEVEL


class CustomLogger:
    def __init__(
        self,
        name: str,
        stream_level: int = DEFAULT_STREAM_LOG_LEVEL,
        file_level: int = logging.DEBUG,
        custom_handle: Path | None = None,
    ):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(max(stream_level, file_level))

        if not getattr(self._logger, "handler_set", None):
            format: str = "%(asctime)s %(levelname)s %(message)s"
            time_format: str = "%Y-%m-%d %H:%M:%S"
            fmt = logging.Formatter(format, time_format)

            console = logging.StreamHandler()
            console.setLevel(stream_level)
            console.setFormatter(fmt)
            self._logger.addHandler(console)

            # --- add the file handler
            if custom_handle is not None:
                log_file: Path = custom_handle
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(fmt)
                file_handler.setLevel(file_level)
                self._logger.addHandler(file_handler)

            # --- don't add more handlers next time
            self._logger.handler_set = True  # type: ignore
            self._logger.propagate = False

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)
