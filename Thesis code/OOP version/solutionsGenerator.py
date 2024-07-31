import math

from util_functions import UtilFunctions
from BruteForceAlgorithm import runAnalysis
from time import time
import itertools
import re
import pandas as pd
from NetworkClass import NetworkDisessembler, NetworkDeltaExtractor, DeltaNetworkMotifAnalyzer, GraphVisualization


if __name__ == '__main__':

    # Define lists of sizes and deltas for testing
    algorithms = ["Nauty"]
    sizes = [70, 80]
    deltas = [0.1, 0.2]

    for n in range(1, 11):
        filename = f'networks/network{n}.txt'
        network = UtilFunctions.csv2network(filename)
        if len(network.keys()) > 100:
            for algo_type in algorithms:
                # Iterate over each delta size
                for d in deltas:
                    delta_size = d

                    # Iterate over each network size
                    for s in sizes:
                        network_size = s
                        # # Generate random solutions based on the network size and delta size
                        solutions_number = 10

                        additional_edges_number = int(network_size * delta_size)

                        # Ensure the number of solutions is feasible by adding additional edges
                        test = math.comb(additional_edges_number, int(network_size * delta_size))
                        while solutions_number**2 > math.comb(additional_edges_number, math.ceil(network_size * delta_size)):
                            additional_edges_number += 2
                            test = math.comb(additional_edges_number, math.ceil(network_size * delta_size))

                        # Slice the network based on the adjusted size
                        network = dict(itertools.islice(network.items(), int(network_size+additional_edges_number)))

                        # Generate random solution set and save them to excel file
                        solution_set = UtilFunctions.generateRandSolutionSet(network, solutions_number, network_size, delta_size)
                        filename_full = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsFull.xlsx'
                        filename_modified = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsModified.xlsx'
                        # filename_full = '/home/ubuntu/PycharmProjects/ThesisCode/Thesis code/Solutions/solutionsFull.xlsx'
                        # filename_modified = '/home/ubuntu/PycharmProjects/ThesisCode/Thesis code/Solutions/solutionsModified.xlsx'
                        col_names = ['edge', 'Activation/Repression', 'delta']
                        UtilFunctions.solutionSetFull2excel(solution_set, col_names, filename_full)
                        UtilFunctions.solutionSetModified2excel(solution_set, col_names, filename_modified)

                        # Test delta extraction on full solutions
                        n = 3
                        filename = filename_full
                        solutions = UtilFunctions.excel2solutionSetList(filename)
                        analyses_full = []
                        time_full = []
                        for i, sol in enumerate(solutions):
                            start_time = time()
                            # Perform motif analysis the original solutions
                            analyzer = DeltaNetworkMotifAnalyzer(sol, n, algo_type)
                            analysis = analyzer.analyze(sol)
                            analyses_full.append(analysis)
                            # Save analysis results to CSV files
                            filename = f'Analyses/network{n}{algo_type}_delta{delta_size}_size{network_size}_Analysis{i}_full.csv'
                            analyzer.saveAnalysis(analysis, filename)
                            # analyzer.saveAnalysis(analyzer.originAnalysis, filename)
                            end_time = time()
                            elapsed_time = end_time - start_time
                            time_full.append(elapsed_time)
                        ave_time = sum(time_full)/len(time_full)
                        print(f"\n##ALGORITHM USED {algo_type}##")
                        print(f'delta = {d}, network size = {s}')
                        print(f'Average time full analysis took was {ave_time} seconds')
                        print('Time for each full analysis below:')
                        print(time_full)

                        # Test delta extraction on modified solutions
                        n = 3
                        filename = filename_modified
                        solutions = UtilFunctions.excel2solutionSetList(filename)
                        analyses_modified = []
                        time_modified = []
                        for i, sol in enumerate(solutions):
                            start_time = time()
                            if i == 0:
                                # Perform motif analysis the original solution
                                analyzer = DeltaNetworkMotifAnalyzer(sol, n, algo_type)
                                analysis = analyzer.originAnalysis
                            else:
                                # Extract delta network from modified solutions
                                extractor = NetworkDeltaExtractor(n, sol)
                                extractor.extractDeltaNetwork()
                                delta = extractor.getDeltaNetwork()
                                deltaNetwork = NetworkDisessembler(delta).getNetwork()
                                # Perform motif analysis on the delta network
                                analysis = analyzer.analyze(deltaNetwork)
                                analysis = analyzer.compare(sol, deltaNetwork, analysis)
                                end_time = time()
                                elapsed_time = end_time - start_time
                                time_modified.append(elapsed_time)

                            # Save analysis results to CSV files
                            filename = f'Analyses/network{n}{algo_type}_delta{delta_size}_size{network_size}_Analysis{i}_modified.csv'
                            analyses_modified.append(analysis)
                            analyzer.saveAnalysis(analysis, filename)

                        ave_time = sum(time_modified)/len(time_modified)
                        print(f"\n##ALGORITHM USED {algo_type}##")
                        print(f'delta = {d}, network size = {s}')
                        print(f'Average time modified analysis took was {ave_time} seconds')
                        print('Time for each modified analysis below:')
                        print(time_modified)
                    #
            # ## Compare full analyses vs. modified analyses
            # for i in range(0, solutions_number):
            #     print('### ANALYSIS NO.' + str(i))
            #     analysis_full = analyses_full[i]
            #     analysis_modified = analyses_modified[i]
            #     for row_num, row_full in analysis_full.iterrows():
            #         row_modified = analysis_modified.loc[row_num]
            #         motif_full = row_full['Motif']
            #         motif_modified = row_modified['Motif']
            #         if motif_full != motif_modified:
            #             print(f'Motif is not the same in row {row_num}')
            #             print('Full:')
            #             print(motif_full)
            #             print('modified')
            #             print(motif_modified)
            #         num_full = row_full['Number of appearances in network']
            #         num_modified = row_modified['Number of appearances in network']
            #         if num_full != num_modified:
            #             print(f'Number of appearances is not the same in row {row_num}')
            #             print('modified is missing:')
            #             print(num_full-num_modified)
            #         indices_full = set(row_full['Edges indices'])
            #         indices_modified = set(row_modified['Edges indices'])
            #         if indices_full != indices_modified:
            #             print(f'Indices list is not the same in Motif No. {row_num}')
            #             if indices_full.difference(indices_modified):
            #                 print('Missing locations in modified analysis:')
            #                 print(indices_full.difference(indices_modified))
            #             if indices_modified.difference(indices_full):
            #                 print('Missing locations in full analysis:')
            #                 print(indices_modified.difference(indices_full))

            # motifs_df = runAnalysis(3, solutions[2])
            # new_network = UtilFunctions.addMotifs2Network(solutions[2], motifs_df)
            # ## Draw combination graph for all solutions
            # GraphDrawer = GraphVisualization(solutions)
            # GraphDrawer.createCombinationGraph(solutions, f"Graph-Combined_solutions-Network_size={s}")
            # GraphDrawer.createDeltaNetworkGraph(solutions[2], f"Graph-Delta-Network_size={s}_solution2")
            # GraphDrawer.createRegularGraph(solutions[2], f"Graph-Regular-Network_size={s}_solution2")
            # GraphDrawer.createMotifDeltaNetworkGraph(new_network, f"Graph-Motif_delta-Network_size={s}_solution2", 1)
            # GraphDrawer.createMotifNetworkGraph(new_network, f"Graph-Motif_reg-Network_size={s}_solution2", 1)










