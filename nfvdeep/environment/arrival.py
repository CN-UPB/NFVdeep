import json
import heapq
import logging
import random
import numpy as np
from abc import abstractmethod
from collections.abc import Generator
from nfvdeep.environment.sfc import ServiceFunctionChain


class ArrivalProcess(Generator):
    """Abstract class that implements the request generation. Inheriting classes implement the parameterization of to be generated SFCs."""

    def __init__(self, **kwargs):
        self.timeslot = 1

        # generate requests according to abstract method `generate_requests`
        self.requests = self.generate_requests()

        # order arriving SFCs according to their arrival time
        self.requests = [((sfc.arrival_time, num), sfc)
                         for num, sfc in enumerate(self.requests)]
        heapq.heapify(self.requests)

    def send(self, arg):
        """Implements the generator interface. Generates a list of arriving SFC requests for the
        respective timeslot starting from timeslot one.
        """
        req = []

        if len(self.requests) <= 0:
            self.throw()

        # add SFCs to the batch until the next SFC has an arrival time that exceeds the internal timestep
        while len(self.requests) > 0 and self.requests[0][1].arrival_time <= self.timeslot:
            _, sfc = heapq.heappop(self.requests)
            req.append(sfc)

        # increment internal timeslot
        self.timeslot += 1
        return req

    def throw(self, type=None, value=None, traceback=None):
        """Raises an Error iff all SFCs were already generated."""
        raise StopIteration

    @staticmethod
    def factory(config):
        """Factory method to allow for an easier generation of different arrival processes."""

        if 'type' not in config:
            arrival = JSONArrivalProcess(config)

        arrival_type = config['type']
        params = {key: value for key, value in config.items() if key != 'type'}

        # set a seed solely if the episodes should be static, i.e. should always get the same requests for each episode
        if not params['static']:
            params['seed'] = None

        if arrival_type == 'poisson_arrival_uniform_load':
            arrival = PoissonArrivalUniformLoad(**params)

        else:
            raise ValueError('Unknown arrival process')

        return arrival

    @abstractmethod
    def generate_requests(self):
        """Abstract method that must generate a list of SFCs to be emitted."""
        raise NotImplementedError('Must be overwritten by an inheriting class')


class JSONArrivalProcess(ArrivalProcess):
    def __init__(self, request_path):
        """Instantiate an arrival process that generates SFC requests at their respective arrival timeslot from a specified JSON file."""
        assert(isinstance(request_path, str))
        assert(request_path.endswith('.json'))
        self.request_path = request_path
        super().__init__()

    def generate_requests(self):
        """Generates SFC objects according to their paramterization as given by a JSON file."""
        # load SFCs from specified JSON file
        with open(self.request_path, 'rb') as file:
            requests = json.load(file)

        def parse_vnfs(vnfs): return [tuple(vnf.values()) for vnf in vnfs]

        # create SFC objects with parameters specified in the JSON file
        req = []
        for sfc in requests:
            vnfs = parse_vnfs(sfc.pop('vnfs'))
            sfc = ServiceFunctionChain(vnfs=vnfs, **sfc)

            req.append(sfc)

        return req


class StochasticProcess(ArrivalProcess):
    def __init__(self, num_requests, load_generator):
        self.num_requests = num_requests
        self.load_generator = load_generator

        super().__init__()

    def generate_requests(self):
        load_gen = self.load_generator.next_sfc_load()
        arrival_gen = self.next_arrival()

        req = []
        while len(req) < self.num_requests:
            arrival_time, ttl = next(arrival_gen)
            sfc_params = next(load_gen)
            sfc = ServiceFunctionChain(
                arrival_time=arrival_time, ttl=ttl, **sfc_params)
            req.append(sfc)

        
        return req


class PoissonArrivalUniformLoad(StochasticProcess):
    def __init__(self, num_timeslots, num_requests, service_rate, num_vnfs, sfc_length, bandwidth, max_response_latency, cpus, memory, vnf_delays, seed=None, **kwargs):
        if not seed is None:
            random.seed(seed)

        # generate random parameters for episode
        self.num_requests = random.randint(*num_requests)
        self.num_timeslots = random.randint(*num_timeslots)

        # derive parametrisation of arrival- & service-time distribution
        self.mean_arrival_rate = self.num_requests / self.num_timeslots
        self.mean_service_rate = random.randint(*service_rate)

        # generate SFC & VNF parameters uniformly at random within their bounds
        load_generator = UniformLoadGenerator(
            sfc_length, num_vnfs, max_response_latency, bandwidth, cpus, memory, vnf_delays)

        super().__init__(self.num_requests, load_generator)

    def next_arrival(self):
        # interarrival times conform to an exponential distribution with the rate parameter `mean_service_rate`
        arrival_times = [random.expovariate(
            self.mean_arrival_rate) for _ in range(self.num_requests)]
        arrival_times = np.ceil(np.cumsum(arrival_times))
        arrival_times = arrival_times.astype(int)

        # service times conform to an exponential distribution with the rate parameter `1 / mean_service_rate`
        service_times = [random.expovariate(
            1 / self.mean_service_rate) for _ in range(len(arrival_times))]
        service_times = np.floor(service_times)
        service_times = service_times.astype(int)

        for arrival_time, service_time in zip(arrival_times, service_times):
            yield arrival_time, service_time


class UniformLoadGenerator:
    def __init__(self, sfc_length, num_vnfs, max_response_latency, bandwidth, cpus, memory, vnf_delays):
        # SFC object generation parameters
        self.sfc_length = sfc_length
        self.num_vnfs = num_vnfs
        self.bandwidth = bandwidth
        self.max_response_latency = max_response_latency

        # VNF object generation parameters
        self.cpus = cpus
        self.memory = memory
        self.vnf_delays = vnf_delays

    def next_sfc_load(self):
        # generate requests demands for `num_vnfs` distinct types of VNFs
        vnf_types = [(random.randint(*self.cpus), random.uniform(*self.memory))
                     for _ in range(self.num_vnfs)]
        delays = [random.uniform(
            *self.vnf_delays) for _ in range(self.num_vnfs)]

        # create SFC objects with uniform loads
        while True:
            sfc_params = {}

            # randomly choose VNFs that compose the SFC
            num_sfc_vnfs = random.randint(*self.sfc_length)
            vnfs_idx = [random.randint(0, len(vnf_types) - 1)
                        for _ in range(num_sfc_vnfs)]

            # generate all VNFs comprised in the SFC
            sfc_params['vnfs'] = [vnf_types[idx] for idx in vnfs_idx]
            sfc_params['processing_delays'] = [delays[idx] for idx in vnfs_idx]
            sfc_params['max_response_latency'] = random.uniform(
                *self.max_response_latency)
            sfc_params['bandwidth_demand'] = random.uniform(*self.bandwidth)

            yield sfc_params
