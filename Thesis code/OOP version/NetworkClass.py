from main import runAnalysis
import pandas as pd

class Network:

    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def getNodes(self):
        return self.nodes.values()

    def getEdges(self):
        return self.edges.values()

    def addNode(self, node):
        self.nodes.setdefault(node.getName(), node)
        # edges = node.getEdges()
        # for edge in edges:
        #     self.edges.setdefault(edge.getIndex(), edge)

    def addEdge(self, edge):
        self.edges.setdefault(edge.getIndex(), edge)
        # nodes = edge.getNodes()
        # for node in nodes:
        #     self.nodes.setdefault(node.getName(), node)

    def createNode(self, name):
        if self.isNodeIn(name):
            return self.nodes[name]
        else:
            return Node(name)

    def createEdge(self, index, edge):
        nodeA_name = edge[0][0]
        nodeB_name = edge[0][1]
        nodeA = self.createNode(nodeA_name)
        nodeB = self.createNode(nodeB_name)
        edgeObject = Edge(index, nodeA, nodeB, edge[1], edge[2])
        nodeA.addEdge(edgeObject)
        nodeB.addEdge(edgeObject)
        self.addNode(nodeA)
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

    def isNodeIn(self, nodeName):
        if nodeName in self.nodes.keys():
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
                    currentDelta.transferNetwork(self.deltaNetwork)

    def getDeltaNetwork(self):
        return self.deltaNetwork

    def getkDistanceNodeDown(self, node, k, visited):
        # Base Case
        if node is None or k < 0:
            return visited

        # Add current node to visited
        visited.addNode(node)

        # If we didn't reach k distance --> recur for neighbors
        if k > 0:
            # Add edges of current node to visited
            for e in node.getEdges():
                visited.addEdge(e)
            for neighbor in node.getNeighbors():
                if neighbor.getName() not in visited.nodes.keys():
                    self.getkDistanceNodeDown(neighbor, k - 1, visited)
        return visited


class DeltaNetworkMotifAnalyzer:

    def __init__(self, originNetwork, motif_size):
        self.originNetwork = originNetwork
        self.motif_size = motif_size
        self.originAnalysis = self.analyze(self.originNetwork)

    def analyze(self, network):
        df = runAnalysis(self.motif_size, network)
        return df

    def saveAnalysis(self, df, filename):
        df.to_csv(filename)

    def compare(self, network, analysis):
        originAnalysis_copy = self.originAnalysis.copy()

        if network:

            for row_num, row in originAnalysis_copy.iterrows():
                origin_indices_remove = []
                motifs = row['Edges indices']
                for index, motif in enumerate(motifs):
                    for edge in motif:
                        delta = network[edge][2]
                        if delta == -1:
                            origin_indices_remove.append(index)
                            break
                # Update data in original analysis
                originAnalysis_copy.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(motifs) if idx not in origin_indices_remove]
                originAnalysis_copy.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx not in origin_indices_remove]
                originAnalysis_copy.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(origin_indices_remove)

            for row_num, row in analysis.iterrows():
                network_indices_keep = []
                motifs = row['Edges indices']
                for index, motif in enumerate(motifs):
                    for edge in motif:
                        delta = network[edge][2]
                        if delta == 1:
                            network_indices_keep.append(index)
                            break
                # Update data in analysis
                analysis.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(motifs) if idx in network_indices_keep]
                analysis.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx in network_indices_keep]
                analysis.at[row_num, 'Number of appearances in network'] = len(network_indices_keep)

                if analysis['Number of appearances in network'].loc[row_num] > 0:
                    if row_num in originAnalysis_copy.index:
                        originAnalysis_copy.at[row_num, 'Edges indices'].extend(analysis.at[row_num, 'Edges indices'])
                        originAnalysis_copy.at[row_num, 'Location of appearances in network'].extend(analysis.at[row_num, 'Location of appearances in network'])
                        originAnalysis_copy.at[row_num, 'Number of appearances in network'] += len(network_indices_keep)
                    else:
                        originAnalysis_copy.append(analysis.loc[row_num], inplace=True)
        return originAnalysis_copy












