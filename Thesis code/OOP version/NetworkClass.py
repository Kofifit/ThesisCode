from main import runAnalysis
import pandas as pd

class Network:
    """
    Represents a network of nodes and edges. The network can add nodes, edges,
    retrieve information about nodes and edges, and perform operations to manipulate
    its structure.
    """

    def __init__(self):
        """
        Initializes a Network object with empty nodes and edges dictionaries.
        """
        self.nodes = dict()  # Dictionary to store nodes
        self.edges = dict()  # Dictionary to store edges

    def getNodes(self):
        """
        Returns a list of all nodes in the network.
        """
        return self.nodes.values()

    def getEdges(self):
        """
        Returns a list of all edges in the network.
        """
        return self.edges.values()

    def addNode(self, node):
        """
        Adds a node to the network.

        Parameters:
            node (Node): Node object to be added to the network.
        """
        self.nodes.setdefault(node.getName(), node)

    def addEdge(self, edge):
        """
        Adds an edge to the network.

        Parameters:
            edge (Edge): Edge object to be added to the network.
        """
        self.edges.setdefault(edge.getIndex(), edge)

    def createNode(self, name):
        """
        Creates a new node if it doesn't exist already.

        Parameters:
            name (str): Name of the node to be created.

        Returns:
            Node: Newly created or existing Node object.
        """
        if self.isNodeIn(name):
            return self.nodes[name]
        else:
            return Node(name)

    def createEdge(self, index, edge):
        """
        Creates a new edge and its associated nodes if they don't exist already in the network.

        Parameters:
            index (int): Index of the edge.
            edge (list): List containing edge information: [node names, function, delta].
        """
        # Get node names from the edge
        nodeA_name = edge[0][0]
        nodeB_name = edge[0][1]
        # Create node object for each node
        nodeA = self.createNode(nodeA_name)
        nodeB = self.createNode(nodeB_name)
        # Create edge object using its nodes
        edgeObject = Edge(index, nodeA, nodeB, edge[1], edge[2])
        # Add the edge to the nodes objects
        nodeA.addEdge(edgeObject)
        nodeB.addEdge(edgeObject)
        # Add nodes to the network
        self.addNode(nodeA)
        self.addNode(nodeB)
        # Add edge to the network
        self.addEdge(edgeObject)

    def extractEdges(self):
        """
        Retrieve all edges from the nodes in the network and then add them to the edges of the network
        """
        for node in self.nodes.values():
            for edge in node.getEdges():
                self.addEdge(edge)

    def extractNodes(self):
        """
        Retrieve all nodes from the edges in the network and then add them to the nodes of the network
        """
        for edge in self.edges.values():
            for node in edge.getNodes():
                self.addNode(node)

    def isEdgeIn(self, edgeIndex):
        """
        Checks if an edge exists in the network.

        Parameters:
            edgeIndex (int): Index of the edge to check.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        if edgeIndex in self.edges.keys():
            return True
        return False

    def isNodeIn(self, nodeName):
        """
        Checks if a node exists in the network.

        Parameters:
            nodeName (str): Name of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        if nodeName in self.nodes.keys():
            return True
        return False

    def transferNetwork(self, network):
        """
        Transfers nodes and edges from one network to another.

        Parameters:
            network (Network): Another network object to transfer nodes and edges to.
        """
        for nodeName, node in self.nodes.items():
            network.addNode(node)
        for edgeIndex, edge in self.edges.items():
            network.addEdge(edge)



class Edge:
    """
    Represents an edge connecting two nodes in a network. An edge contains information
    about its source and target nodes, associated function, and delta value.
    """

    def __init__(self, index, sourceNode, targetNode, function, delta):
        """
        Initializes an Edge object with source node, target node, function, and delta.

        Parameters:
            index (int): Index of the edge.
            sourceNode (Node): Source node of the edge.
            targetNode (Node): Target node of the edge.
            function (str): Associated function of the edge.
            delta (int): Delta value of the edge.
        """
        self.index = index
        self.sourceNode = sourceNode
        self.targetNode = targetNode
        self.nodes = [sourceNode, targetNode] # List of nodes connected by the edge
        self.function = function
        self.delta = delta

    def getIndex(self):
        """
        Returns the index of the edge.
        """
        return self.index

    def getNodes(self):
        """
        Returns a list of nodes connected by the edge.
        """
        return self.nodes

    def isDelta(self):
        """
        Check if edge appears in the original network.
        If the edge does not appear in the original network, but appears in this network the delta would be 1.
        """
        if self.delta == 1 or self.delta == -1:
            return True
        return False


class Node:
    """
    Represents a node in a network. A node contains information about its name,
    neighboring nodes, and connected edges.
    """

    def __init__(self, name):
        """
        Initializes a Node object with a name and empty neighbors and edges dictionaries.

        Parameters:
            name (str): Name of the node.
        """

        self.name = name  # Name of the node
        self.neighbors = dict()  # Dictionary to store neighboring nodes
        self.edges = dict()  # Dictionary to store connected edges
        self.neighbor_edge_table = dict()  # Dictionary to store neighbor-edge pairs

    def getName(self):
        """
        Returns the name of the node.
        """
        return self.name

    def getNeighbors(self):
        """
        Returns a list of neighboring nodes.
        """
        return self.neighbors.values()

    def getEdges(self):
        """
        Returns a list of edges connected to the node.
        """
        return self.edges.values()

    def getEdgeByNeighbor(self, neighborName):
        """
        Returns an edge object by node name
        :param nodeName:
        :return:
        """
        return self.neighbor_edge_table[neighborName]

    def addEdge(self, edge):
        """
        Adds an edge to the node's edges dictionary and update neighboring nodes.

        Parameters:
            edge (Edge): Edge object to be added.
        """
        self.edges.setdefault(edge.getIndex(), edge)
        for node in edge.getNodes():
            if node.getName() != self.name:
                self.neighbors.setdefault(node.getName(), node)
                self.neighbor_edge_table.setdefault(node.getName(), edge)


class NetworkEssembler:
    """
    Assembles a new network object from a network that appear in a dictionary format.
    It creates a new network object and adds edges and nodes from the provided network.
    """

    def __init__(self, network):
        """
        Initializes a NetworkEssembler object and creates a network from a given network.

        Parameters:
            network (dict): Network in a dictionary format.
        """
        self.network = self.createNetwork(network)

    def getNetwork(self):
        """
        Returns the assembled network.
        """
        return self.network

    def createNetwork(self, network):
        """
        Creates a network from a given network in a dictionary format.

        Parameters:
            network (dict): Network in a dictionary format.

        Returns:
            Network: Assembled network object.
        """
        networkObject = Network()
        for index, edge in network.items():
            networkObject.createEdge(index, edge)
        return networkObject

class NetworkDisessembler:
    """
    Disassembles a network object into a network in a dictionary format.
    It creates a new network in a dictionary format and adds edges and nodes from the provided network.
    """

    def __init__(self, objectNetwork):
        """
         Initializes a NetworkDisessembler object and creates a network in a dictionary format from a given network
         object.

         Parameters:
             objectNetwork (Network): Network object to be disassembled.
         """
        self.network = self.createNetwork(objectNetwork)

    def getNetwork(self):
        """
        Returns the disassembled network.
        """
        return self.network

    def createNetwork(self, objectNetwork):
        """
        Creates a network in a dictionary format from a given network object.

        Parameters:
            objectNetwork (Network): Network object to be disassembled.

        Returns:
            dict: Disassembled network in a dictionary format.
        """
        network = dict()
        for edge in objectNetwork.getEdges():
            network[edge.getIndex()] = [[edge.sourceNode.getName(), edge.targetNode.getName()], edge.function, edge.delta]
        return network


class NetworkDeltaExtractor:
    """
    Extracts the delta network from the main network based on delta edges.
    The delta network contains the interactions that appear in the current network and not in the original network.
    In addition, it contains the areas around these new interactions based on the motif size.
    """

    def __init__(self, motif_size, network):
        """
        Initializes a NetworkDeltaExtractor object with a motif size and a network, and creates an empty delta network.

        Parameters:
            motif_size (int): Size of the motif for delta extraction.
            network (Network): Main network object.
        """
        self.motif_size = motif_size  # Motif size for delta extraction
        self.network = NetworkEssembler(network).getNetwork()  # Assemble the network
        self.deltaNetwork = Network()  # Initialize an empty delta network

    def extractDeltaNetwork(self):
        """
        Extracts the delta network from the main network based on delta edges.
        Use the 'getkDistanceNodeDown' function to explore and retrieve the nodes around the delta edges
        """
        edges = self.network.getEdges()
        for edge in edges:
            if edge.isDelta():
                for node in edge.getNodes():
                    visited = Network()
                    currentDelta = self.getkDistanceNodeDown(node, self.motif_size - 2, visited)
                    currentDelta.transferNetwork(self.deltaNetwork)

    def getDeltaNetwork(self):
        """
         Returns the extracted delta network.

         Returns:
             Network: Extracted delta network.
         """
        return self.deltaNetwork

    def getkDistanceNodeDown(self, node,  k, visited):
        """
        Recursively explores nodes down to a certain distance from the given node.
        This method is used to extract a sub-network (delta network) by exploring nodes
        down to a specified distance based on a given motif size.

        Parameters:
            node (Node): Current node to explore.
            k (int): Remaining distance to explore.
            visited (Network): Network object to store visited nodes and edges.

        Returns:
            Network: Network containing visited nodes and edges within the specified distance.
        """
        # Base Case
        if node is None or k < 0:
            return visited

        # Add current node to visited
        visited.addNode(node)

        # If there are remaining distance to explore
        if k > 0:
            # Add edges of the current node to the visited network
            for e in node.getEdges():
                if e.delta != -1:
                    visited.addEdge(e)
            # Recursively explore neighboring nodes
            for neighbor in node.getNeighbors():
                edge = node.getEdgeByNeighbor(neighbor.getName())
                # Only explore unvisited neighboring nodes
                if neighbor.getName() not in visited.nodes.keys() and edge.delta != -1:
                    self.getkDistanceNodeDown(neighbor, k - 1, visited)
        return visited


class DeltaNetworkMotifAnalyzer:
    """
    Analyzes delta network motifs and compares them with the original network.
    """

    def __init__(self, originNetwork, motif_size):
        """
        Initializes a DeltaNetworkMotifAnalyzer object with the original network and motif size.

        Parameters:
            originNetwork (Network): Original network to analyze.
            motif_size (int): Size of the motif for analysis.
        """
        self.originNetwork = originNetwork  # Original network for analysis
        self.motif_size = motif_size  # Motif size for analysis
        self.originAnalysis = self.analyze(self.originNetwork)  # Perform analysis on the original network

    def analyze(self, network):
        """
        Performs motif analysis on the given network.

        Parameters:
            network (Network): Network to perform analysis on.

        Returns:
            DataFrame: DataFrame containing analysis results.
        """
        df = runAnalysis(self.motif_size, network)
        return df

    def saveAnalysis(self, df, filename):
        """
        Saves analysis results to a CSV file.

        Parameters:
            df (DataFrame): DataFrame containing analysis results.
            filename (str): Name of the CSV file to save the analysis results.
        """

        df.to_csv(filename)

    def compare(self, network, delta_network, analysis):
        """
        Compares the analysis results between the original network and a modified network.
        This method identifies the common motifs for the modified network and the original,
        and adds these common motifs to the modified network analysis.

        Parameters:
            network (dict): Modified network for comparison.
            analysis (DataFrame): DataFrame containing analysis results for the modified network.

        Returns:
            DataFrame: DataFrame containing the compared analysis results.
        """
        originAnalysis_copy = self.originAnalysis.copy()
        submotif_remove = []

        # Iterate over each row (motif) in the analysis of the modified network
        for row_num, row in analysis.iterrows():
            network_indices_remove = []
            motifs = row['Edges indices']
            # Iterate over each motif's location in the analysis of the modified network
            for index, motif in enumerate(motifs):
                remove = False
                sub_motif = False
                # Check if the edge appears in the modified network (delta == 1)
                for edge in motif:
                    delta = network[edge][2]
                    if delta == -1:
                        remove = True
                        break
                    elif delta == 1:
                        sub_motif = True
                # If the edge is found in the modified network, keep it in the modified analysis
                if remove:
                    network_indices_remove.append(index)
                # If exists a sub-motif, add it to the list of sub-motifs for removal from origin analysis
                elif not remove and sub_motif:
                    submotif_remove.append(tuple(i for i in motif if network[i][2] == 0))

            # Update the analysis to keep edges that were found in the modified network
            analysis.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(motifs) if idx not in network_indices_remove]
            analysis.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx not in network_indices_remove]
            analysis.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(network_indices_remove)

        # Iterate over each row (motif) in the original analysis
        for row_num, row in originAnalysis_copy.iterrows():
            origin_indices_keep = []
            motifs = row['Edges indices']
            # Iterate over each motif's location in the analysis of the original network
            for index, motif in enumerate(motifs):
                keep = False
                for edge in motif:
                    delta = network[edge][2]
                    # Check if the edge does not appear in the delta network
                    if edge not in delta_network.keys():
                        # Keep motif in the original analysis
                        keep = True
                    # Check if the edge does not appear in the modified network at all
                    if delta == -1:
                        # Remove motif from the original analysis
                        keep = False
                        break
                if keep:
                    origin_indices_keep.append(index)

            # Update the original analysis to remove edges that were not found in the modified network
            originAnalysis_copy.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(motifs) if idx in origin_indices_keep]
            originAnalysis_copy.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx in origin_indices_keep]
            originAnalysis_copy.at[row_num, 'Number of appearances in network'] = len(origin_indices_keep)

            # If the motif still has appearances in the analysis of the modified network, add them to the copy original analysis
            if analysis['Number of appearances in network'].loc[row_num] > 0:
                if row_num in originAnalysis_copy.index:
                    originAnalysis_copy.at[row_num, 'Edges indices'].extend(analysis.at[row_num, 'Edges indices'])
                    originAnalysis_copy.at[row_num, 'Location of appearances in network'].extend(analysis.at[row_num, 'Location of appearances in network'])
                    originAnalysis_copy.at[row_num, 'Number of appearances in network'] += analysis.at[row_num, 'Number of appearances in network']
                else:
                    originAnalysis_copy = originAnalysis_copy._append(analysis.loc[row_num])

        temp_analysis = originAnalysis_copy

        # Get all indices for each motif
        all_motif_indices = []
        for motif_num, motif in temp_analysis.iterrows():
            all_motif_indices.append(motif['Edges indices'])

        # Fin all motifs in the analysis that are truly sub-motifs
        submotif_remove = [[] for i in range(0, len(temp_analysis))]
        for motif_type_index, motif_indices in enumerate(all_motif_indices):
            for motif_loc_index, motif in enumerate(motif_indices):
                if self.is_submotif(motif, [x for xs in all_motif_indices for x in xs]):
                    submotif_remove[motif_type_index].append(motif_loc_index)

        # Remove all sub-motifs from the analysis
        for row_num, indices_remove in enumerate(submotif_remove):
            if indices_remove:
                row = temp_analysis.iloc[row_num]
                temp_analysis.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(row['Edges indices']) if idx not in indices_remove]
                temp_analysis.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx not in indices_remove]
                temp_analysis.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(indices_remove)

        final_analysis = temp_analysis

        return final_analysis

    def is_submotif(self, subgraph_indices, all_indices):
        for graph in all_indices:
            if len(graph) > len(subgraph_indices):
                if set(subgraph_indices).issubset(set(graph)):
                    return True
        return False















