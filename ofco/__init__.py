import logging
from .main import *

__author__ = "Florian Aymanns"
__email__ = "florian.aymanns@epfl.ch"

#####################################################################
# Adrian Sager 15/11/2021:
#     Added logger
#
#####################################################################
def initialize():
    if '_ofco_initialized' in globals():
        return
    global _ofco_initialized
    _ofco_initialized = True

    logger = logging.getLogger('ofco')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = \
        logging.Formatter('[%(asctime)s] %(levelname).1s T%(thread)d %(filename)s:%(lineno)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

initialize()