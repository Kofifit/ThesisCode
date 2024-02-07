from main import runAnalysis

class Network:

    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def getNodes(self):
        return self.nodes.values()

    def getEdges(self):
        return self.edges.values()

    def addNode(self, node):
        self.nodes[node.getName()] = node
        # edges = node.getEdges()
        # for edge in edges:
        #     self.edges.setdefault(edge.getIndex(), edge)

    def addEdge(self, edge):
        self.edges.setdefault(edge.getIndex(), edge)
        # nodes = edge.getNodes()
        # for node in nodes:
        #     self.nodes.setdefault(node.getName(), node)

    def createEdge(self, index, edge):
        nodeA = Node(edge[0][0])
        nodeB = Node(edge[0][1])
        edgeObject = Edge(index, nodeA, nodeB, edge[1], edge[2])
        nodeA.addEdge(edgeObject)
        self.addNode(nodeA)
        nodeB.addEdge(edgeObject)
        self.addNode(nodeB)
        self.addEdge(edgeObject)

    def extractEdges(self):
        for node in self.nodes.values():
            for edge in node.getEdges():
                self.addEdge(edge)

    def extractNodes(self):
        for edge in self.edges.values():
            for node in edge.getNodes():
                self.addNode(node)

    def isEdgeIn(self, edgeIndex):
        if edgeIndex in self.edges.keys():
            return True
        return False

    def transferNetwork(self, network):
        for nodeName, node in self.nodes.items():
            network.nodes.setdefault(nodeName, node)
        for edgeIndex, edge in self.edges.items():
            network.edges.setdefault(edgeIndex, edge)



class Edge:

    def __init__(self, index, sourceNode, targetNode, function, delta):
        self.index = index
        self.sourceNode = sourceNode
        self.targetNode = targetNode
        self.nodes = [sourceNode, targetNode]
        self.function = function
        self.delta = delta

    def getIndex(self):
        return self.index

    def getNodes(self):
        return self.nodes

    def isDelta(self):
        if self.delta == 1:
            return True
        return False


class Node:

    def __init__(self, name):
        self.name = name
        self.neighbors = dict()
        self.edges = dict()

    def getName(self):
        return self.name

    def getNeighbors(self):
        return self.neighbors.values()

    def getEdges(self):
        return self.edges.values()

    def addEdge(self, edge):
        self.edges[edge.getIndex()] = edge
        for node in edge.getNodes():
            if node.getName() != self.name:
                self.neighbors[node.getName()] = node


class NetworkEssembler:

    def __init__(self, network):
        self.network = self.createNetwork(network)

    def getNetwork(self):
        return self.network

    def createNetwork(self, network):
        networkObject = Network()
        for index, edge in network.items():
            networkObject.createEdge(index, edge)
        return networkObject

class NetworkDisessembler:

    def __init__(self, objectNetwork):
        self.network = self.createNetwork(objectNetwork)

    def getNetwork(self):
        return self.network

    def createNetwork(self, objectNetwork):
        network = dict()
        for edge in objectNetwork.getEdges():
            network[edge.getIndex()] = [[edge.sourceNode.getName(), edge.targetNode.getName()], edge.function, edge.delta]
        return network


class NetworkDeltaExtractor:

    def __init__(self, motif_size, network):
        self.motif_size = motif_size
        self.network = NetworkEssembler(network).getNetwork()
        self.deltaNetwork = Network()


    def extractDeltaNetwork(self):
        edges = self.network.getEdges()
        for edge in edges:
            if edge.isDelta():
                for node in edge.getNodes():
                    visited = Network()
                    currentDelta = self.getkDistanceNodeDown(node, self.motif_size - 2, visited)
                    currentDelta.extractNodes()
                    currentDelta.transferNetwork(self.deltaNetwork)

    def getDeltaNetwork(self):
        self.deltaNetwork.extractEdges()
        return self.deltaNetwork

    def getkDistanceNodeDown(self, node, k, visited):
        # Base Case
        if node is None or k < 0:
            return visited
        # Add current node to visited
        visited.addNode(node)
        # If we didn't reach k distance --> recur for neighbors
        if k > 0:
            for neighbor in node.getNeighbors():
                if neighbor.getName() not in visited.nodes.keys():
                    self.getkDistanceNodeDown(neighbor, k - 1, visited)
        return visited


class DeltaNetworkMotifAnalyzer:

    def __init__(self, originNetwork, motif_size):
        self.originNetwork = originNetwork
        self.motif_size = motif_size
        self.originAnalysis = self.analyze(self.originNetwork, 'originAnalysis')

    def analyze(self, network, filename):
        df = runAnalysis(self.motif_size, network, filename)
        return df

    def compare(self, network, filename):
        analysis = self.analyze(network, filename)
        originAnalysis_copy = self.originAnalysis.copy(deep=True)
        indices_remove = []
        for row_num, indices in enumerate(self.originAnalysis['Edges indices']):
            for index in indices:
                delta = network[index][2]
                if delta == -1:
                    indices_remove.append(row_num)
                    break
        originAnalysis_copy.drop(indices_remove, inplace=True)
        indices_keep = []
        for row_num, indices in enumerate(analysis['Edges indices']):
            for index in indices:
                delta = network[index][2]
                if delta == 1:
                    indices_keep.append(row_num)
                    break












