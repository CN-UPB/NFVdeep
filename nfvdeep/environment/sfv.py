class ServiceFunctionChain:
    def __init__(self, arrival_time, ttl, bandwidth_demand, max_response_latency, vnfs, processing_delays=None):
        '''Creating a new service function chain

        Args:
          arrival_time (int): Arrival time of the SFC request
          ttl (int): Time to live of the SFC
          bandwidth_demand (float): The minimal ammount of bandwidth that is acceptable
          max_response_latency (float): The maximal acceptable latency
          vnfs (list of tuples): A (ordered) list of all Vnfs of the SFC
          processing_delays (list of float): A (ordered) list of all delays of the Vnfs (default: no delays)'''

        self.arrival_time = arrival_time
        self.ttl = ttl
        self.bandwidth_demand = bandwidth_demand
        self.max_response_latency = max_response_latency
        self.vnfs = vnfs
        self.num_vnfs = len(self.vnfs)

        self.processing_delays = [
            0 for _ in self.vnfs] if processing_delays is None else processing_delays

    def __repr__(self):
        """String representation of the SFC instance."""
        s = ' '.join([str([vnf for vnf in self.vnfs])])
        return s
