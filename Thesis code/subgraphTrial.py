import numpy as np
import itertools
import pandas as pd


def subgraphGenerator(n):
    # This function get an integer (n) as the input
    # This function gives back all the fully-connected sub-graphs of the size n.

    ## Get all possible edges ##

    # Define a list of the vertices
    nodes = np.arange(0, n)

    # Define list to store all possible edges
    edgeAll = np.zeros((n ** 2 - n, 2), dtype=int)

    # Loop to find all possible edges
    index = 0
    for n1 in range(0, n):
        for n2 in range(0, n):
            if n1 != n2:
                edgeAll[index, 0] = n1
                edgeAll[index, 1] = n2
                index += 1

    ## Find all possible edge combinations that contain all vertices

    # Define counter for loop
    counter = 0

    # Define an empty list for the combinations
    combo = list()

    # Define an empty list for the number of edges for each sub-graph
    edgeNum = list()

    # For loop to get all combinations
    for conNum in range(n - 1, n ** 2 - n + 1):

        # Get all possible combinations with itertools package
        currentCombo = list(itertools.combinations(np.arange(n ** 2 - n), conNum))

        # Loop to find if each combination has all the vertices
        for c in currentCombo:
            lst = np.unique(np.concatenate((edgeAll[c,:]), axis=0))

            # If not, the combination is removed from the list
            if not np.array_equal(lst, nodes):
                currentCombo.remove(c)

        # Add current list of combinations to the general list of combinations
        combo.extend(currentCombo)
        edgeNum.extend(np.repeat(conNum, len(currentCombo)))

    return combo, edgeNum, edgeAll


def motifGeneratorBrute(n):
    combo, edgeNumList, edgeAll = subgraphGenerator(n)

    motifs = list()
    possEdgeNum = np.unique(edgeNumList)
    possCombination = list(itertools.permutations(range(n)))
    matched = []

    # Go over each number of possible edges
    for edgeNum in possEdgeNum:

        # Get a list of indices where number of edges is equal to edgeNum
        indices = [idx for idx, val in enumerate(edgeNumList) if val == edgeNum]

        # Get the first motif from the list of indices
        for motifIndex in indices:

            # Check to see if sub-graph was already matched
            if motifIndex in matched:
                continue

            # Add motif index to matched list and to output
            matched.append(motifIndex)
            motifs.append(combo[motifIndex])

            # Get the motif as graph for example [[0,1],[0,2]]
            motif = edgeAll[combo[motifIndex], :]
            possOrder = list(itertools.permutations(range(len(motif))))
            motifList = []
            for order in possOrder:
                m = []
                for i in order:
                    m.extend(motif[i])
                motifList.append(m)

            # Go over all other indices to find sub-graphs that match the motif
            for otherIndex in indices:

                # Check to see if sub-graph was already matched
                if otherIndex in matched:
                    continue

                # Get the sub-graph as a graph and make a copy
                otherGraph = edgeAll[combo[otherIndex], :]
                graphList = []
                for edge in otherGraph: graphList.extend(edge)
                copyGraph = graphList.copy()

                # Get a list of the nodes in the sub-graph in order (For example [[2,1],[2,0]] --> [2,1,0])
                nodes = uniqueNodes(graphList)

                # Replace the "names" of the nodes to find if it matches the motif
                for com in possCombination:
                    for index, val in enumerate(graphList):
                        i = [idx for idx, node in enumerate(nodes) if node == val]
                        copyGraph[index] = com[i[0]]
                    if copyGraph in motifList:
                        matched.append(otherIndex)
                        break

    return motifs, edgeAll


def networkCombosBrute(n, network):
    # This function gets two inputs - a network (a list of lists) and a node number (an integer)
    # It returns a table with the possible motifs, the number of times they appeared in the given network
    # and the locations where they appeared
    # The algorithm used to find the motifs and their location in the network is Brute-Force algorithm

    # Run the motif generator function and define empty lists to stored the location of the sub-graphs
    # and the number of times each motif appeared in the network
    motifs, edgeAll = motifGeneratorBrute(n)
    counters = np.zeros(len(motifs), dtype=int)
    locations = [[] for i in range(len(motifs))]
    matched = []
    motifsOutput = []

    edgeNum = np.arange(n - 1, n ** 2 - n + 1)  # all possible numbers of edges
    edgeNetNum = len(network)  # Find number of edges in the network
    combo = []  # Define an empty list for the combinations
    possCombination = list(itertools.permutations(range(n)))  # Get a list of all possible combination of named nodes

    # Find all sub-graphs with n nodes in the network
    for num in edgeNum:
        currentCombo = list(itertools.combinations(np.arange(edgeNetNum), num))

        # Loop to find if each combination has exactly n nodes ###### PROBLEM!!!!! cutting the iterating list
        for c in currentCombo:
            lst = np.unique(np.concatenate((network[c, :]), axis=0))
            nodeNum = len(lst)

            # If the combination contains only 3 nodes it is added to the list
            if np.array_equal(nodeNum, n):
                combo.append(c)


    # Compare the sub-graph in the network to the motifs
    for m_index, motif in enumerate(motifs):

        # Get the motif as graph for example [[0,1],[0,2]]
        mGraph = edgeAll[motif, :]
        motifsOutput.append(mGraph)
        possOrder = list(itertools.permutations(range(len(mGraph))))
        motifList = []
        for order in possOrder:
            m = []
            for i in order:
                m.extend(mGraph[i])
            motifList.append(m)

        # Loop through all sub-graphs and match with current motif
        for sg_index, sub_graph in enumerate(combo):

            # Check if number of edges of sub is not greater than motif's
            if len(sub_graph) > len(motif) or sg_index in matched:
                continue

            # Get the sub-graph as a graph and make a copy
            subGraph = network[sub_graph, :]
            graphList = []
            for edge in subGraph: graphList.extend(edge)
            copyGraph = graphList.copy()

            # Get a list of nodes in the sub-graph
            nodes = uniqueNodes(graphList)

            # Replace the "names" of the nodes to find if it matches the motif
            for com in possCombination:
                for index, val in enumerate(graphList):
                    i = [idx for idx, node in enumerate(nodes) if node == val]
                    copyGraph[index] = com[i[0]]
                if copyGraph in motifList:
                    counters[m_index] += 1
                    locations[m_index].append(subGraph)
                    break

    # Add motifs, counters and locations into a dataframe and return as output
    output = pd.DataFrame(
        {'Motif': motifsOutput,
         'Counter': counters,
         'Locations': locations
         })

    return output


def uniqueNodes(vector):

    # This functions get a vector of integers
    # It returns the unique values in the vector, in the order they appeared
    # For example: vector = [1,0,0,2] --> the function returns [1,0,2]

    output = []
    for val in vector:
        if val not in output or not output:
            output.append(val)
    return output


while True:
    # Define the network
    network = []

    # Run tests
    for i in range(5):
        edge = [np.random.randint(5), np.random.randint(5)]
        network.append(edge)

    network = np.array(network)
    print(network)

    # Define number of nodes for motifs
    n = 4

    # Run the function
    res = networkCombosBrute(n, network)

    print(res)







# # Define the network
# network = []
# # Open network text file and load it into list of edges
# f = open('/Users/renanabenyehuda/Desktop/Thesis code/network.txt', 'r')
#
# # Run over each line and get the edge
# for line in f:
#     nodes = line.split()
#     edge = [int(nodes[0]), int(nodes[1])]
#     network.append(edge.copy())
#
# network = np.array(network)
# Define number of nodes for motifs
# n = 3
#
# # Run the function
# res = networkCombosBrute(n, network)
#
# print(res)

