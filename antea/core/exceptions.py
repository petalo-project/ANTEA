"""
Define ANTEA-specific exceptions
"""

class ANTEAException(Exception):
    """ Base class for ANTEA exceptions hierarchy """

class NoInputFiles(ANTEAException):
    """ Input files list is not defined """

class WaveformEmptyTable(ANTEAException):
    pass

class ParameterNotSet(ANTEAException):
    pass
