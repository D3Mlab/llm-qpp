import logging
import sys

def setup_logging(name: str, config = {}, level = None, output_file = None) -> logging.Logger:
    """
    Setup logging configuration and return a logger instance.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Levels are 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL). If not provided, it is read from the config.
        disabled (bool, optional): Whether the logger is disabled.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    #prevents duplicates if a class has multiple instances
    if (logger.hasHandlers()):
        logger.handlers.clear()


    
    #check if logger is disabled
    logging_config = config.get('logging',{})
    logger.disabled = logging_config.get('disabled', None)

    if logger.disabled == True:
        return logger

    #Check if level argument is provided, else read from config
    if level == None:
        level = logging.getLevelName(logging_config.get('level','INFO'))

    logger.setLevel(level)

    # Create a log message formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a StreamHandler that outputs log messages to the console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    # Create a StreamHandler that outputs log messages to output file, if it exists
    if not output_file:
        output_file = logging_config.get('log_file') # Returns None if the log_file is missing
    
    if output_file:
        oh = logging.FileHandler(output_file)
        oh.setFormatter(formatter)
        oh.setLevel(level)
        logger.addHandler(oh)
    
    return logger