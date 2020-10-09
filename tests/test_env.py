from environment.network import Network
from environment.sfv import ServiceFunctionChain
from environment.env import Env
from environment.arrival import ArrivalProcess
import networkx as nx


def test_network():
    '''test if the network works with a cycle graph'''
    G = nx.Graph()
    G.add_node(0, cpu=3, memory=10.0, bandwidth=20.0)
    G.add_node(1, cpu=3, memory=30.0, bandwidth=20.0)
    G.add_node(2, cpu=4, memory=10.0, bandwidth=20.0)
    G.add_node(3, cpu=4, memory=30.0, bandwidth=20.0)
    G.add_node(4, cpu=8, memory=50.0, bandwidth=1.0)
    G.add_node(5, cpu=8, memory=5.0, bandwidth=30.0)
    G.add_node(6, cpu=2, memory=50.0, bandwidth=30.0)

    G.add_edge(0, 1, latency=50.0)
    G.add_edge(1, 2, latency=1.0)
    G.add_edge(2, 3, latency=1.0)
    G.add_edge(3, 4, latency=1.0)
    G.add_edge(4, 5, latency=1.0)
    G.add_edge(5, 6, latency=50.0)
    G.add_edge(6, 0, latency=1.0)

    network = Network(G)
    resources = network.calculate_resources()
    assert resources[0] == {'cpu': 3, 'memory': 10.0, 'bandwidth': 20.0}
    assert resources[5] == {'cpu': 8, 'memory': 5.0, 'bandwidth': 30.0}

    assert not network.check_vnf_resources((2, 50.0), 20.0, 0)
    assert not network.check_vnf_resources((6, 30.0), 20.0, 6)
    #assert not network.check_vnf_resources((1, 1.0), 50.0, 6)
    assert not network.check_vnf_resources((60, 300.0), 200.0)

    # ServiceFunctionChain(arrivaltime, ttl, bandwidth, latency, vnfs, processing_delays)
    sfc1 = ServiceFunctionChain(0, 3, 20, 3, [(2, 30.0), (2, 30.0)], [1, 0])

    # check if resources gets updated properly
    assert network.check_embeddable(sfc1)
    assert not network.embed_vnf(sfc1, (2, 30.0), 0)
    assert network.embed_vnf(sfc1, (2, 30.0), 1)
    assert not network.embed_vnf(sfc1, (2, 30.0), 1)
    assert network.embed_vnf(sfc1, (2, 30.0), 3)

    assert network.calculate_current_latency(sfc1) == 3

    resources = network.calculate_resources()
    assert resources[1] == {'cpu': 1, 'memory': 0.0, 'bandwidth': 0.0}
    assert resources[3] == {'cpu': 2, 'memory': 0.0, 'bandwidth': 0.0}
    assert resources[5] == {'cpu': 8, 'memory': 5.0, 'bandwidth': 30.0}

    # latency should be enough
    assert network.check_sfc_constraints(sfc1)

    network.update(3)
    # sfc is still active - resources should not change
    resources = network.calculate_resources()
    assert resources[1] == {'cpu': 1, 'memory': 0.0, 'bandwidth': 0.0}
    assert resources[3] == {'cpu': 2, 'memory': 0.0, 'bandwidth': 0.0}
    assert resources[5] == {'cpu': 8, 'memory': 5.0, 'bandwidth': 30.0}

    # sfc expires - resources should be back to normal
    network.update()
    resources = network.calculate_resources()
    assert resources[1] == {'cpu': 3, 'memory': 30.0, 'bandwidth': 20.0}
    assert resources[3] == {'cpu': 4, 'memory': 30.0, 'bandwidth': 20.0}
    assert resources[5] == {'cpu': 8, 'memory': 5.0, 'bandwidth': 30.0}

    # ServiceFunctionChain(arrivaltime, ttl, bandwidth, latency, vnfs, processing_delays)
    sfc1 = ServiceFunctionChain(0, 3, 1, 5, [(1, 1.0), (2, 2.0)])
    sfc2 = ServiceFunctionChain(0, 3, 2, 5, [(1, 1.0), (2, 2.0)], [5, 4])

    assert network.embed_vnf(sfc1, (1, 1.0), 1)
    assert network.embed_vnf(sfc1, (2, 2.0), 5)
    assert network.calculate_current_latency(sfc1) == 4
    assert network.embed_vnf(sfc2, (1, 1.0), 5)
    assert network.calculate_current_latency(sfc2) == 5
    assert network.embed_vnf(sfc2, (2, 2.0), 6)

    resources = network.calculate_resources()
    assert resources[1] == {'cpu': 2, 'memory': 29.0, 'bandwidth': 19.0}
    assert resources[5] == {'cpu': 5, 'memory': 2.0, 'bandwidth': 27.0}
    assert resources[6] == {'cpu': 0, 'memory': 48.0, 'bandwidth': 28.0}

    assert network.check_sfc_constraints(sfc1)
    # latency is not enough for sfc2:
    assert network.calculate_current_latency(sfc2) == 59
    assert not network.check_sfc_constraints(sfc2)

    network.update(5)
    # check if bandwidth is updated corretly
    sfc1 = ServiceFunctionChain(0, 3, 1, 5, [(1, 1.0), (2, 2.0), (1, 1.0)])
    sfc2 = ServiceFunctionChain(0, 3, 2, 5, [(1, 1.0), (1, 2.0), (2, 2.0)])

    assert network.embed_vnf(sfc1, (1, 1.0), 1)
    assert network.embed_vnf(sfc1, (2, 2.0), 5)
    assert network.embed_vnf(sfc1, (1, 1.0), 5)
    assert network.calculate_current_latency(sfc1) == 4
    assert network.embed_vnf(sfc2, (1, 1.0), 1)
    assert network.embed_vnf(sfc2, (1, 2.0), 1)
    assert network.embed_vnf(sfc2, (2, 2.0), 5)

    resources = network.calculate_resources()
    assert resources[1] == {'cpu': 0, 'memory': 26.0, 'bandwidth': 17.0}
    assert resources[5] == {'cpu': 3, 'memory': 0.0, 'bandwidth': 27.0}
