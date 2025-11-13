from dataclasses import dataclass, field
from typing import Any, Final

###################### TAG CLASSES FOR COMMUNICATION ######################

class TAGS:
    __rdict__ = {}
    DATA_ADJ_MATRIX: Final[int] = 0
    DATA_GOAL: Final[int] = 1
    DATA_RUN_REPORT: Final[int] = 2
    DATA_MISC: Final[int] = 3
    INFO_TIME_ESTIMATION: Final[int] = 10
    INFO_SURVIVAL: Final[int] = 11
    INFO_ABNORMAL: Final[int] = 12
    INFO_MISC: Final[int] = 12

class SURVIVAL:
    __rdict__ = {}
    HOST_RUNNING: Final[int] = 0
    HOST_NORMAL_FINISHED: Final[int] = 1
    HOST_ABNORMAL_SHUTDOWN: Final[int] = 2

class REASONS:
    __rdict__ = {}
    REACH_MAX_ITER: Final[int] = 0
    HARD_TIMEOUT: Final[int] = 1
    FAKE_RESULT: Final[int] = 2

## Generation revert dictionary for tag classes
## such that you could reverse map the tag to its name
def init_rdict(c):
    for k, v in c.__dict__.items():
        if not k.startswith('__'):
            c.__rdict__[v] = k

init_rdict(TAGS)
init_rdict(REASONS)
init_rdict(SURVIVAL)

###################### DATACLASS FOR RECORDING MPI RUNNING STATUS ######################

@dataclass
class AGENT_STATUS:
    assigned_job: Any = None
    estimation_time: float = None
    current_iter: int = None
    tik_time: int = 0
    up_time: float = 0
    abnormal_counter: int = 0

    def __str__(self) -> str:
        if self.assigned_job:
            t = f'Current assigned_job = {self.assigned_job}, estimation_time = {self.estimation_time}, current_iter = {self.current_iter}. \n' \
                f'Current tik_time = {self.tik_time}, real up_time = {self.up_time}, abnormal_counter = {self.abnormal_counter}. \n'
        else:
            t = f'Current no job assigned. \n' \
                f'Current tik_time = {self.tik_time}, real up_time = {self.up_time}, abnormal_counter = {self.abnormal_counter}. \n'
        return t
    
@dataclass
class INDIVIDUAL_STATUS:
    assigned: list[int] = field(default_factory=list)
    repeated: int = 0
    finished: bool = False
    minimal_estimation_time: float = 1e9

    def __str__(self) -> str:
        if self.assigned:
            t = f'Individual {self.individual.scope} has been repeated {self.repeated} times,\n' \
                f'it is currently been assign in agent rank = {self.assigned},\n' \
                f'Minimal estimation time for this individual is {self.minimal_estimation_time}. \n'
        else:
            t = f'Individual {self.individual.scope} has finish with {self.repeated} repeation times.\n' \
                'Waiting to be assigned. \n'
            
        return t


###################### UTILS ######################

class DUMMYINDV:
    pass

def DUMMYFUNC(*args, **kwds):
    pass

## the formal function parser for yaml that avoid eval() troubles
from importlib import import_module
def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)
