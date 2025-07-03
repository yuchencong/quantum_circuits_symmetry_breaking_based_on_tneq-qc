import time, mpi4py
import numpy as np
import functools, itertools
from mpi_generation import Generation, Individual
from mpi_core import TAGS, REASONS, DUMMYFUNC, AGENT_STATUS, SURVIVAL
from callbacks import CALLBACKS
from collections import deque

class MPI_Overlord():

    ## OVERLORD SEND:
    ## 1. survival ping
    ## 2. job data

    ## OVERLORD RECEIVE
    ## 1. survival info
    ## 2. abnormal when receiving job
    ## 3. estimation info
    ## 4. job result report

    ## SINGLE JOB PIPELINE
    ## 1. overlord match an agent and an individual
    ## 2. overlord ask the individual to submit the data
    ## 3. overlord isend the data to agent
    ## 4. overlord receive result and report to the individual


    ################ UTILS ################
    def __init__(self, comm: mpi4py.MPI.COMM_WORLD, **kwds) -> None:
        self.kwds = kwds
        self.logger = kwds['logger']
        self.time = 0
        self.comm = comm
        self.agent_size = self.comm.Get_size() - 1
        self.host_status = SURVIVAL.HOST_RUNNING

        # generation
        self.max_generation = kwds['experiment']['max_generation']
        self.collection_of_generations = []

        # agents
        self.agent_report_buffer = deque()
        self.available_agents = dict(
            itertools.zip_longest(list(range(1, self.agent_size+1)), [], fillvalue=AGENT_STATUS()))

    def tik_and_sleep(self, sec):
        self.time += sec
        time.sleep(sec)

    def call_with_interval(self, func, interval):
        if self.time % interval == 0:
            return func
        else:
            return DUMMYFUNC
        
    ################ MPI COMMUNICATION ################
    def sync_goal(self):
        goal = self.kwds['experiment']['evoluation_goal']
        self.evoluation_goal = np.load(goal)
        self.evoluation_goal = self.comm.bcast(self.evoluation_goal, root=0)

    def process_msg_surv(self, msg):
        rank_surv = msg['rank']

        self.available_agents[rank_surv].tik_time = msg['time']
        self.available_agents[rank_surv].up_time = msg['real_up_time']
        self.available_agents[rank_surv].current_iter = msg['current_iter']

        if msg['busy']:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'current completion rate {msg["current_iter"]} / {msg["max_iter"]}.')
        else:
            self.logger.info(f'Received survival report from agent rank {rank_surv}, ' \
                            f'reported tik time {msg["time"]}, real up time {msg["real_up_time"]},' \
                            f'not working currently.')

        return

    def process_msg_estm(self, msg):
        rank_estm = msg['rank']

        self.available_agents[rank_estm].estimation_time = msg['required_time']

        self.logger.info(f'Received estimation report from agent rank {rank_estm}, ' \
                        f'required time is about {msg["required_time"]} with current loss {msg["loss"]}.')

        return

    def process_msg_abnm(self, msg):
        rank_abnm = msg

        self.available_agents[rank_abnm].abnormal_counter += 1

        self.logger.info(f'Received abnormal report from agent rank {rank_abnm}, ' \
                         f'current abnormal count for agent rank {rank_abnm} is ' \
                         f'{self.available_agents[rank_abnm].abnormal_counter}.')

        return

    def process_msg_rept(self, msg):

        ## Lazy process 
        ## forward the message to a deque
        ## then ask the generation to process it later

        self.agent_report_buffer.append(msg)
        rank_rept = msg['rank']

        self.available_agents[rank_rept].assigned_job = None
        self.available_agents[rank_rept].estimation_time = None
        self.available_agents[rank_rept].current_iter = None

        self.logger.info(f'Received final report from agent rank {rank_rept} with reason {REASONS[msg["reason"]]}, ' \
                        f'reported final iteration {msg["current_iter"]}, ' \
                        f'final loss {msg["loss"]}.')

    def broadcast_finish(self):
        ## by default the while loop will change the status to normal shutdown
        ## therefore once the while loop is broken, should change it to abnormal
 
        if self.host_status == SURVIVAL.HOST_RUNNING:
            self.host_status = SURVIVAL.HOST_ABNORMAL_SHUTDOWN
        self.call_agent_survivability()
        return

    def call_agent_survivability(self):
        for agent in self.available_agents.keys():
            self.comm.isend(self.host_status, dest=agent, tag=TAGS.INFO_SURVIVAL)
        return

        
    def tik_and_collect_everything_from_agent(self, sec):
        ## The tik function is a while loop checking the irecv status

        self.time += sec
        start_time = time.time()
        while time.time() - start_time < sec:

            status_surv, msg_surv = self.req_surv.test()
            if status_surv:
                self.process_msg_surv(msg_surv)
                self.req_surv = self.comm.irecv(tag=TAGS.INFO_SURVIVAL)

            status_estm, msg_estm = self.req_estm.test()
            if status_estm:
                self.process_msg_surv(msg_estm)
                self.req_estm = self.comm.irecv(tag=TAGS.INFO_TIME_ESTIMATION)

            status_abnm, msg_abnm = self.req_abnm.test()
            if status_abnm:
                self.process_msg_abnm(msg_abnm)
                self.req_abnm = self.comm.irecv(tag=TAGS.INFO_ABNORMAL)

            status_rept, msg_rept = self.req_rept.test()
            if status_rept:
                self.process_msg_rept(msg_rept)
                self.req_rept = self.comm.irecv(tag=TAGS.DATA_RUN_REPORT)

        return

    ################ GENERATION OPERATION ################
    def get_current_generation(self):
        return self.collection_of_generations[-1]

    def despatch_generation_suggest_agent(self):
        pass

    def match_and_assign_job_to_agent(self):
        self.__check_available_agent__()
        if len(self.available_agents)>0:
            for agent in self.available_agents:
                self.current_generation.distribute_indv(agent)

    def despatch_generation_collect_result(self):
        while self.agent_report_buffer:
            r = self.agent_report_buffer.popleft()
            self.get_current_generation.collect_indv_report(r)
        return

    def span_generation(self):
        if not len(self.collection_of_generations):
            self.collection_of_generations.append(Generation(name='generation_init', **self.kwds))
            return True

        if len(self.collection_of_generations) >= self.max_generation:
            return False
        
        else:
            ## is_finished now is TOTALLY finished, including evaluation and evolution
            current_generation = self.collection_of_generations[-1]
            if current_generation.is_finished():
                next_generation = Generation(current_generation,
                    name=f'generation_{len(self.collection_of_generations)+1:03d}', **self.kwds)
                self.collection_of_generations.append(next_generation)

            return True

    def generate_picklable_instance(self):
        pass

    def __call__(self):
        ## when the overload is called
        ## 1. initilize 4 mpi recv comm
        ## 2. sync goal with the rank = 0
        ## 3. entering the main generation spanning loop,
        ##    different from the former version,
        ##    now the overlord only send the messages from agent to generation,
        ##    the generation will deal with that.
        ## 4. clean comm and send finish msg to all the agents

        self.req_surv = self.comm.irecv(tag=TAGS.INFO_SURVIVAL)
        self.req_estm = self.comm.irecv(tag=TAGS.INFO_TIME_ESTIMATION)
        self.req_abnm = self.comm.irecv(tag=TAGS.INFO_ABNORMAL)
        self.req_rept = self.comm.irecv(tag=TAGS.DATA_RUN_REPORT)

        self.sync_goal()
        while self.span_generation():
            cg = self.get_current_generation()
            self.call_with_interval(self.check_available_agent, 4)
            self.call_with_interval(self.assign_job, 4)
            self.call_with_interval(self.__collect_result__, 4)
            self.call_with_interval(self.__report_agents__, 180)
            self.call_with_interval(self.__report_generation__, 160)
            self.tik_and_collect_everything_from_agent(2)
        else:
            CALLBACKS.OVERLOAD()
            self.host_status = SURVIVAL.HOST_NORMAL_FINISHED
            

        
        self.req_estm.Cancel();self.req_estm.Free()
        self.req_surv.Cancel();self.req_surv.Free()
        self.req_abnm.Cancel();self.req_abnm.Free()
        self.req_rept.Cancel();self.req_rept.Free()
        self.broadcast_finish()

if __name__ == '__main__':
    pipeline = Overlord
    pipeline()