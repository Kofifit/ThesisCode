class Network:

    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def addEdge(self, index, edge):
        nodeA, nodeB, edgeObject = self.createEdge(index, edge)
        self.edges[edgeObject.getIndex()] = edgeObject
        self.nodes[nodeA.getName()] = nodeA
        self.nodes[nodeB.getName()] = nodeB

    def createEdge(self, index, edge):
        nodeA = Node(edge[0][0])
        nodeB = Node(edge[0][1])
        edgeObject = Edge(index, [nodeA, nodeB], edge[1], edge[2])
        nodeA.addEdge(edgeObject)
        nodeB.addEdge(edgeObject)
        return nodeA, nodeB, edgeObject

class Edge:

    def __init__(self, index, nodes, function, delta):
        self.index = index
        self.nodes = nodes
        self.function = function
        self.delta = delta

    def getIndex(self):
        return self.index

    def getNodes(self):
        return self.nodes


class Node:

    def __init__(self, name):
        self.name = name
        self.neighbors = dict()

    def getName(self):
        return self.name

    def getNeighbors(self):
        return self.neighbors

    def addEdge(self, edge):
        for node in edge.getNodes():
            if node.getName() != self.name:
                self.neighbors[node.getName()] = node


class NetworkBuilder:

    def __init__(self, network):
        self.network = self.createNetwork(network)

    def getNetwork(self):
        return self.network

    def createNetwork(self, network):
        networkObject = Network()
        for index, edge in network.items():
            networkObject.addEdge(index, edge)
        return networkObject


class NetworkDeltaExtractor:

    def __init__(self, motif_size, network):
        self.motif_size = motif_size
        self.network = NetworkBuilder(network).getNetwork()





