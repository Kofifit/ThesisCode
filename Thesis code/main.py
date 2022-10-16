import numpy as np
import itertools
from dfs import Graph


def subgraph_generator(n):
    # This function get an integer (n) as the input
    # This function gives back all the fully-connected sub-graphs of the size n.

    # Get all possible edges

    # Define a list of the vertices
    nodes = np.arange(0, n)

    # Define list to store all possible edges
    edge_all = np.zeros((n ** 2 - n, 2), dtype=int)

    # Loop to find all possible edges
    index = 0
    for n1 in range(0, n):
        for n2 in range(0, n):
            if n1 != n2:
                edge_all[index, 0] = n1
                edge_all[index, 1] = n2
                index += 1

    # Find all possible edge combinations that contain all vertices

    # Define an empty list for the combinations
    all_subgraphs = list()

    # Define an empty list for the number of edges for each sub-graph
    edge_num = list()

    if n < 2:
        return all_subgraphs, edge_num

    # For loop to get all combinations
    for number_edges in range(n - 1, n ** 2 - n + 1):

        # Get all possible combinations with itertools package
        all_combinations = list(itertools.combinations(np.arange(n ** 2 - n), number_edges))

        # Loop to find if each combination is fully connected
        for combo in all_combinations:
            subgraph = dict()
            g = Graph()
            nodesList = []
            for edge in combo:
                if edge_all[edge][0] not in subgraph:
                    subgraph[edge_all[edge][0]] = []
                subgraph[edge_all[edge][0]].append(edge_all[edge][1])  # Save subgraph
                g.addEdge(edge_all[edge][0], edge_all[edge][1])  # Save subgraph in Graph class for DFS algorithm
                nodesList.extend([edge_all[edge][0], edge_all[edge][1]])  # Save edges to check if graph contains n nodes
            nodesList = np.array(nodesList)
            nodesList = np.unique(nodesList)
            nodesList = np.sort(nodesList, axis=None)
            if not np.array_equal(nodesList, nodes):
                continue

            if g.DFS(0):
                subgraph['Matched'] = False
                all_subgraphs.append(subgraph)
                edge_num.append(number_edges)

    return all_subgraphs, edge_num


def motif_generator(n):
    all_subgraphs, edge_num = subgraph_generator(n)

    motifs = list()
    possEdgeNum = list(range(n - 1, n ** 2 - n + 1))
    possCombination = list(itertools.permutations(range(0, n)))
    edgesMotifs = []

    # Go over each number of possible edges
    for edgeNum in possEdgeNum:

        # Get a list of indices where number of edges is equal to edgeNum
        indices = [idx for idx, val in enumerate(edge_num) if val == edgeNum]

        # Get the first motif from the list of indices
        for mIndex in indices:

            motif = all_subgraphs[mIndex]

            # Check to see if sub-graph was already matched
            if motif['Matched']:
                continue

            # Add motif index to matched list and to output
            motif['Matched'] = True
            motifs.append(motif)
            edgesMotifs.append(edgeNum)

            # Go over all other indices to find sub-graphs that match the motif
            for oIndex in indices:

                subgraph = all_subgraphs[oIndex]
                # Check to see if sub-graph was already matched
                if subgraph['Matched']:
                    continue

                # Replace the "names" of the nodes to find if it matches the motif
                for combo in possCombination:

                    buffer_subgraph = dict()
                    buffer_subgraph['Matched'] = True
                    for key, value in subgraph.items():
                        if key != 'Matched':
                            buffer_subgraph[combo[key]] = [combo[v] for v in value]

                    for key, value in buffer_subgraph.items():
                        if key != 'Matched':
                            buffer_subgraph[key] = sorted(value)

                    if motif == buffer_subgraph:  # Check if sub-graph matches motif
                        subgraph['Matched'] = True
                        break

    return motifs, edgesMotifs


def motif_finder_in_network(n, network):
    # This function gets two inputs - a network (a list of lists) and a node number (an integer)
    # It returns a table with the possible motifs, the number of times they appeared in the given network
    # and the locations where they appeared
    # The algorithm used to find the motifs and their location in the network is Brute-Force algorithm

    # Run the motif generator function and define empty lists to stored the location of the sub-graphs
    # and the number of times each motif appeared in the network
    motifs, edgesMotifs = motif_generator(n)
    counters = np.zeros(len(motifs), dtype=int)
    all_subgraphs = []
    all_nodesLists = []
    edgesSubgraphs = []

    edgeNum = np.arange(n - 1, n ** 2 - n + 1)  # all possible numbers of edges
    edgeNetNum = len(network)  # Find number of edges in the network
    combo = []  # Define an empty list for the combinations
    possCombination = list(itertools.permutations(range(n)))  # Get a list of all possible combination for renaming nodes

    # Find all sub-graphs with n nodes in the network
    for num in edgeNum:
        currentCombo = list(itertools.combinations(np.arange(edgeNetNum), num))

        # Loop to find if each combination has exactly n nodes
        for c in currentCombo:
            lst = np.unique(np.concatenate((network[tuple(c), :]), axis=0))
            nodeNum = len(lst)

            # If the combination contains only 3 nodes it is added to the list
            if np.array_equal(nodeNum, n):
                combo.append(c)

    # Check if subgraph is fully connected and convert type to dictionary
    for c in combo:
        g = Graph()
        subgraph = dict()
        nodesList = []
        for edge in c:
            if network[edge, 0] not in subgraph:
                subgraph[network[edge, 0]] = []
            subgraph[network[edge, 0]].append(network[edge, 1])
            edgesSubgraphs.append(len(c))
            g.addEdge(network[edge, 0], network[edge, 1])  # Save subgraph in Graph class for DFS algorithm
            nodesList.extend([network[edge, 0], network[edge, 1]])  # Save edges to check if graph contains n nodes
        nodesList = np.array(nodesList)
        nodesList = np.unique(nodesList)
        nodesList = np.sort(nodesList, axis=None)
        if len(nodesList) != n:
            continue
        if g.DFS(nodesList[0]):
            subgraph['Matched'] = False
            all_subgraphs.append(subgraph)
            all_nodesLists.append(nodesList)

    # Compare the sub-graphs in the network to the motifs, iterate over motifs
    for m_index, motif in enumerate(motifs):

        # Iterate over all sub-graphs and match with current motif
        for g_index, subgraph in enumerate(all_subgraphs):

            # Check if number of edges of sub is the same and the subgraph wasn't matched before
            if subgraph['Matched'] or edgesSubgraphs[g_index] != edgesMotifs[m_index]:
                continue

            # Replace the "names" of the nodes to find if it matches the motif
            for combo in possCombination:
                combo_dict = {}
                buffer_subgraph = dict()
                buffer_subgraph['Matched'] = True
                for i, node in enumerate(all_nodesLists[g_index]):
                    combo_dict[node] = combo[i]
                for key, value in subgraph.items():
                    if key != 'Matched':
                        buffer_subgraph[combo_dict[key]] = [combo_dict[v] for v in value]

                for key, value in buffer_subgraph.items():
                    if key != 'Matched':
                        buffer_subgraph[key] = sorted(value)

                if motif == buffer_subgraph:  # Check if sub-graph matches motif
                    subgraph['Matched'] = True
                    counters[m_index] += 1
                    break

    return all_subgraphs, motifs, counters


def get_unique_nodes(vector):
    # This functions get a vector of integers
    # It returns the unique values in the vector, in the order they appeared
    # For example: vector = [1,0,0,2] --> the function returns [1,0,2]

    output = []
    for val in vector:
        if val not in output or not output:
            output.append(val)
    return output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    for n in range(1, 5):
        motifs, x = motif_generator(n)
        print('\n')
        print('n =', n, '\ncount =', len(motifs))
        if len(motifs) > 0:
            counter = 0
            for motif in motifs:
                counter += 1
                print('\n#', counter)
                for key, values in motif.items():
                    if key != 'Matched':
                        for value in values:
                            print(key, value)

    network = []
    n = input("Please enter number of nodes for the motif search - ")
    while int(n) < 2:
        n = input("Please enter number of nodes that is 2 or greater - ")
    num_inputs = input("Please enter number of edges in your network - ")
    for i in range(int(num_inputs)):
        edge = input("Please enter an edge for example 9 2 - ")
        edge = [int(i) for i in edge.split()]
        network.append(edge)
    network = np.array(network)
    all_subgraphs, motifs, counters = motif_finder_in_network(int(n), network)
    print('\nTotal number of subgraphs of size n is', len(all_subgraphs), '\nThe subgraphs: ')
    for i, subgraph in enumerate(all_subgraphs):
        print('\n#', i+1)
        for key, values in subgraph.items():
            if key != 'Matched':
                for value in values:
                    print(key, value)
    print('\nn =', n, '\ntotal number of motifs =', len(motifs))
    for i, motif in enumerate(motifs):
        print('\n#', i+1)
        print('count =', counters[i])
        for key, values in motif.items():
            if key != 'Matched':
                for value in values:
                    print(key, value)
