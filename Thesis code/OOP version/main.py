import numpy as np
import itertools

import pandas as pd

from dfs import Graph
from util_functions import UtilFunctions


def subgraph_generator(n):
    # This function get an integer (n) as the input
    # This function gives back all the fully-connected sub-graphs of the size n.

    global edge_all
    edge_all = get_all_possible_edges(n)  # Get a dictionary of all possible edges with n nodes
    all_subgraphs = list()  # Define an empty list for the fully-connected sub-graphs
    edge_num = [0]  # Define an empty list for the number of subgraphs for a specific number of edges

    if n < 2:
        return all_subgraphs, edge_num
    counter = 0
    # For loop to get combinations from n-1 to n^2-n
    for number_edges in range(n - 1, n ** 2 - n + 1):

        # Get all possible combinations with itertools package
        all_combinations = list(itertools.combinations(np.arange(n ** 2 - n), number_edges))

        # Loop to find if each combination is fully connected
        for combo in all_combinations:
            subgraph, g, nodes_subgraph = get_graph(combo, edge_all)
            if not has_n_nodes(n, nodes_subgraph):
                continue
            if is_fully_connected(g):
                subgraph['Matched'] = False
                all_subgraphs.append(subgraph)
                counter += 1

        edge_num.append(counter)

    return all_subgraphs, edge_num


def motif_generator(n):
    all_subgraphs, edge_num = subgraph_generator(n)

    motifs = list()
    possEdgeNum = list(range(n - 1, n ** 2 - n + 1))
    possCombination = list(itertools.permutations(range(0, n)))
    possCombination.pop(0)
    edgesMotifs = []
    i = 1

    # Go over each number of possible edges
    for edgeNum in possEdgeNum:

        # Get a list of indices where number of edges is equal to edgeNum
        indices = np.arange(edge_num[i-1], edge_num[i])
        i += 1

        # Get the first motif from the list of indices
        for mIndex in indices:

            motif = all_subgraphs[mIndex]
            if motif['Matched']:  # Check to see if sub-graph was already matched
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

                    buffer_subgraph = get_buffer_graph(combo, subgraph)
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
    indices = [[] for i in range(0, len(motifs))]
    all_subgraphs = []
    all_nodesLists = []
    edgesSubgraphs = []
    edge_indices_subgraphs = []

    edgeNum = np.arange(n - 1, n ** 2 - n + 1)  # all possible numbers of edges
    edgeNetNum = len(network)  # Find number of edges in the network
    combo = []  # Define an empty list for the combinations
    possCombination = list(itertools.permutations(range(n)))  # Get a list of all possible combination for renaming nodes

    # Get node combinations
    for num in edgeNum:
        current_combinations = list(itertools.combinations(network.keys(), num))
        combo.extend(current_combinations)

    # Add fully connected sub-graphs to the list of sub-graphs
    for c in combo:
        edges = [network[k][0] for k in c]
        nodes = np.unique(np.concatenate(edges, axis=0))
        if has_n_nodes(n, nodes):  # Only if a has at least n nodes, then connectivity would be tested
            subgraph, g, nodes_subgraph = get_graph(c, network)
            if is_fully_connected(g, nodes_subgraph[0]):
                subgraph['Matched'] = False
                all_subgraphs.append(subgraph)
                all_nodesLists.append(np.unique(nodes_subgraph))
                edgesSubgraphs.append(len(c))
                edge_indices_subgraphs.append(c)

    # Compare the sub-graphs in the network to the motifs, iterate over motifs
    for m_index, motif in enumerate(motifs):

        # Iterate over all sub-graphs in the network and match with current motif
        for g_index, subgraph in enumerate(all_subgraphs):
            # Check if subgraph wasn't matched before or number of edges of subgraph isn't the same as the motif
            if subgraph['Matched'] or edgesSubgraphs[g_index] != edgesMotifs[m_index]:
                continue

            for combo in possCombination:
                combo_dict = get_label_dict(all_nodesLists[g_index], combo)
                buffer_subgraph = get_buffer_graph(combo_dict, subgraph)
                if motif == buffer_subgraph:  # Check if sub-graph matches motif
                    subgraph['Matched'] = True
                    subgraph['Motif#'] = m_index
                    counters[m_index] += 1
                    indices[m_index].extend(edge_indices_subgraphs[g_index])
                    break

    return all_subgraphs, motifs, counters, indices


def get_unique_nodes(vector):
    # This function get a vector of integers
    # It returns the unique values in the vector, in the order they appeared
    # For example: vector = [1,0,0,2] --> the function returns [1,0,2]

    output = []
    for val in vector:
        if val not in output or not output:
            output.append(val)
    return output


def get_label_dict(old_labels, new_labels):
    # This function get two lists - old_labels of nodes and new_labels of nodes
    # It returns a dictionary that matches the old labels to the new labels

    label_dict = dict()
    for i, old in enumerate(old_labels):
        label_dict[old] = new_labels[i]
    return label_dict


def get_all_possible_edges(n):
    # This function get an integer n as the number of nodes
    # It returns a dictionary with all possible edges with n nodes
    # ** NOTE all returned edges are positive ([[node, node], positive(1)/negative(2) ]

    # Define dict to store all possible edges
    all_edges = dict()
    # Loop to find all possible edges
    index = 0
    for n1 in range(0, n):
        for n2 in range(0, n):
            if n1 != n2:
                all_edges[index] = [[n1, n2], 1]
                index += 1
    return all_edges


def get_graph(combo, network):
    # This function get a list of edges
    # It returns a subgraph as a dict and as a Graph as well as a list of all nodes in the subgraph

    subgraph = dict()
    g = Graph()
    nodes_subgraph = []
    for edge in combo:
        edge = network[edge][0]
        if edge[0] not in subgraph:
            subgraph[edge[0]] = []
        subgraph[edge[0]].append(edge[1])  # Save subgraph
        g.addEdge(edge[0], edge[1])  # Save subgraph in Graph class for DFS algorithm
        nodes_subgraph.extend([edge[0], edge[1]])
    return subgraph, g, np.array(nodes_subgraph)


def get_buffer_graph(combo, subgraph):
    # This function get a list of new labels for subgraph's nodes and a subgraph
    # It returns a buffer_subgraph with new labeled nodes

    buffer_subgraph = dict()
    buffer_subgraph['Matched'] = True
    for key, val in subgraph.items():
        if key != 'Matched':
            buffer_subgraph[combo[key]] = sorted([combo[v] for v in val])
    return buffer_subgraph


def has_n_nodes(n, nodes_subgraph):
    # This function get a list of nodes and a list of the nodes in a subgraph
    # It returns True if subgraph has all nodes and False otherwise

    nodes_subgraph = np.unique(nodes_subgraph)
    nodes_num = len(nodes_subgraph)
    if np.array_equal(n, nodes_num):
        return True
    else:
        return False


def is_fully_connected(graph, start_node = 0):
    # This function get a Graph and checks if it is fully connected with DFS algorithm
    # It returns True if the graph is fully connected and False otherwise

    return graph.DFS(start_node)

def runAnalysis(n, network, filename):
    
    all_subgraphs, motifs, counters, indices = motif_finder_in_network(int(n), network)
    df = pd.DataFrame()
    df['Motif'] = motifs
    df['Number of appearances in network'] = counters
    indices_col = [set(x) for x in indices]
    df['Edges indices'] = indices_col
    locations = [[] for i in range(0, len(motifs))]
    for subgraph in all_subgraphs:
        temp = []
        for key, values in subgraph.items():
            if key != 'Matched' and key != 'Motif#':
                for value in values:
                    temp.append([key, value])
        locations[subgraph['Motif#']].append(temp)
    df['Location of appearances in network'] = locations
    for i, count in enumerate(counters):
        if count == 0:
            df.drop(i, inplace=True)
    df.to_csv(filename)
    return df


def printAnalysisResult(all_subgraphs, motifs, counters):

    print('\nTotal number of subgraphs of size n is', len(all_subgraphs), '\nThe subgraphs: ')
    for i, subgraph in enumerate(all_subgraphs):
        print('\n#', i + 1)
        for key, values in subgraph.items():
            if key != 'Matched' and key != 'Motif#':
                for value in values:
                    print(key, value)
    print('\nn =', n, '\ntotal number of motifs =', len(motifs))
    for i, motif in enumerate(motifs):
        print('\n#', i + 1)
        print('count =', counters[i])
        for key, values in motif.items():
            if key != 'Matched':
                for value in values:
                    print(key, value)



if __name__ == '__main__':

    n = input("Please enter number of nodes for the motif search - ")
    while int(n) < 2:
        n = input("Please enter number of nodes that is 2 or greater - ")

    choice = input("Network (1) or Solution set (2) - ")
    filename = input('Please enter name of file - ')
    if choice == '1':
        network = UtilFunctions.csv2network(filename)
        network = dict(itertools.islice(network.items(), 20))
        df = runAnalysis(n, network, 'AnalysisResults/analysis.csv')
    elif choice == '2':
        solutions = UtilFunctions.excel2solutionSetList(filename)
        for i, s in enumerate(solutions):
            if len(s) > 50:
                s = dict(itertools.islice(s.items(), 50))
            filename = 'Analysis' + str(i) + '.csv'
            df = runAnalysis(n, s, filename)


