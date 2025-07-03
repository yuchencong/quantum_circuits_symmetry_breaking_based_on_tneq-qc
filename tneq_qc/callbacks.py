from typing import Any
import numpy as np
import joblib

class LOG_FORMATER:
    BLACK_F = "\033[30m {content} \033[0m"
    RED_F = "\033[31m {content} \033[0m"
    GREEN_F = "\033[32m {content} \033[0m"
    YELLOW_F = "\033[33m {content} \033[0m"
    BLUE_F = "\033[34m {content} \033[0m"
    PURPLE_F = "\033[35m {content} \033[0m"
    AZURE_F = "\033[36m {content} \033[0m"
    WHITE_F = "\033[37m {content} \033[0m"

    BLACK_B = "\033[40;37m {content} \033[0m"
    RED_B = "\033[41;37m {content} \033[0m"
    GREEN_B = "\033[42;37m {content} \033[0m"
    YELLOW_B = "\033[43;37m {content} \033[0m"
    BLUE_B = "\033[44;37m {content} \033[0m"
    PURPLE_B = "\033[45;37m {content} \033[0m"
    AZURE_B = "\033[46;37m {content} \033[0m"
    WHITE_B = "\033[47;30m {content} \033[0m"

class CALLBACKS:

    ### Callback functions are now automately called for each domain
    ### Callbacks are not called by initilization function,
    ### which calls all methods that defined (including do_nothing)

    class INDIVIDUAL:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('logger', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}.')
                    ff(*args, **kwds)

    class GENERATION:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        @staticmethod
        def score_summary(generation, logger):
            logger.info('===== {} ====='.format(generation.name))

            for k, v in generation.societies.items():
                logger.info('===== SOCIETY {} ====='.format(k))

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('logger', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}.')
                    ff(*args, **kwds)


    class OVERLOAD:

        @staticmethod
        def do_nothing(*args, **kwds):
            pass

        def record_experiment(self, *args, **kwds):
            with open('experiment.joblib', 'wb') as f:
                joblib.dump(self, f)
            
            logger = kwds.get('logger', None)
            if logger:
                logger.info(f'Experiment saved.')

        def __init__(self, *args: Any, **kwds: Any) -> None:
            logger = kwds.get('logger', None)
            for f in dir(self):
                if not f.startswith('__'):
                    ff = eval(f'self.{f}')
                    if logger:
                        logger.info(f'Calling callback function {ff}')
                    ff(*args, **kwds)