import math

from util_functions import UtilFunctions
from main import runAnalysis
from time import time
import itertools
import re
import pandas as pd
from NetworkClass import NetworkDisessembler, NetworkDeltaExtractor, DeltaNetworkMotifAnalyzer


if __name__ == '__main__':
    # network_filename = 'network.csv'
    # network = UtilFunctions.csv2network(network_filename)
    # network = dict(itertools.islice(network.items(), 60))
    # motifs_df = runAnalysis(3, network, 'temp.csv')
    # new_network = UtilFunctions.addMotifs2Network(network, motifs_df)
    # col_names = ['edge', 'Activation/Repression', 'delta']
    # for i, row in motifs_df.iterrows():
    #     name = 'Motif #' + str(i)
    #     col_names.append(name)
    # UtilFunctions.network2csv('networkWithMotifs.csv', new_network, col_names)
    # n = 3
    # filename = 'Solutions/solutionsModified.xlsx'
    # new_filename = 'Solutions/solutionsModifiedWithMotifs.xlsx'
    # solutions = UtilFunctions.excel2solutionSetList(filename)
    # all_col_names = []
    # solution_set = []
    # for i, s in enumerate(solutions):
    #     if len(s) > 50:
    #         s = dict(itertools.islice(s.items(), 50))
    #     filename = 'Analysis' + str(i) + '.csv'
    #     motifs_df = runAnalysis(n, s, filename)
    #     new_network = UtilFunctions.addMotifs2Network(s, motifs_df)
    #     col_names = ['edge', 'Activation/Repression', 'delta']
    #     for motif_number, row in motifs_df.iterrows():
    #         name = 'Motif #' + str(motif_number)
    #         col_names.append(name)
    #     all_col_names.append(col_names)
    #     solution_set.append(new_network)
    # UtilFunctions.solutionSetFull2excel(solution_set, new_filename, all_col_names)



    # Define lists of sizes and deltas for testing
    sizes = [20]
    deltas = [0.2]

    # Iterate over each delta size
    for d in deltas:
        delta_size = d
        print(f'\n##### TEST FOR DELTA SIZE {delta_size} #####')

        # Iterate over each network size
        for s in sizes:
            network_size = s
            print(f'\n#### TEST FOR NETWORK WITH {network_size} INTERACTIONS ####\n')

            # # Generate random solutions based on the network size and delta size
            solutions_number = 10
            # filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/network.csv'
            # network = UtilFunctions.csv2network(filename)
            # additional_edges_number = int(network_size * delta_size)
            #
            # # Ensure the number of solutions is feasible by adding additional edges
            # test = math.comb(additional_edges_number, int( network_size * delta_size))
            # while solutions_number**2 > math.comb(additional_edges_number, math.ceil(network_size * delta_size)):
            #     additional_edges_number += 2
            #     test = math.comb(additional_edges_number, math.ceil(network_size * delta_size))
            #
            # # Slice the network based on the adjusted size
            # network = dict(itertools.islice(network.items(), int(network_size+additional_edges_number)))
            #
            # # Generate random solution set and save them to excel file
            # solution_set = UtilFunctions.generateRandSolutionSet(network, solutions_number, network_size, delta_size)
            # filename_full = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsFull.xlsx'
            # filename_modified = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsModified.xlsx'
            # col_names = ['edge', 'Activation/Repression', 'delta']
            # UtilFunctions.solutionSetFull2excel(solution_set, col_names, filename_full)
            # UtilFunctions.solutionSetModified2excel(solution_set, col_names, filename_modified)

            # Test delta extraction on full solutions
            n = 3
            filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsFull.xlsx'
            solutions = UtilFunctions.excel2solutionSetList(filename)
            analyses_full = []
            time_full = []
            for i, s in enumerate(solutions):
                start_time = time()
                # Perform motif analysis the original solutions
                analyzer = DeltaNetworkMotifAnalyzer(s, n)
                analysis = analyzer.originAnalysis
                analyses_full.append(analysis)
                # Save analysis results to CSV files
                filename = f'Analyses/delta{delta_size}_size{network_size}_Analysis{i}_full.csv'
                analyzer.saveAnalysis(analyzer.originAnalysis, filename)
                end_time = time()
                elapsed_time = end_time - start_time
                time_full.append(elapsed_time)
            ave_time = sum(time_full)/len(time_full)
            print(f'Average time full analysis took was {ave_time} seconds')
            print('Time for each full analysis below:')
            print(time_full)

            # Test delta extraction on modified solutions
            n = 3
            filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsModified.xlsx'
            solutions = UtilFunctions.excel2solutionSetList(filename)
            analyses_modified = []
            time_modified = []
            for i, s in enumerate(solutions):
                start_time = time()
                if i == 0:
                    # Perform motif analysis the original solution
                    analyzer = DeltaNetworkMotifAnalyzer(s, n)
                    analysis = analyzer.originAnalysis
                else:
                    # Extract delta network from modified solutions
                    extractor = NetworkDeltaExtractor(n, s)
                    extractor.extractDeltaNetwork()
                    delta = extractor.getDeltaNetwork()
                    deltaNetwork = NetworkDisessembler(delta).getNetwork()
                    # Perform motif analysis on the delta network
                    analysis = analyzer.analyze(deltaNetwork)
                    analysis = analyzer.compare(s, analysis)
                    end_time = time()
                    elapsed_time = end_time - start_time
                    time_modified.append(elapsed_time)

                # Save analysis results to CSV files
                filename = f'Analyses/delta{delta_size}_size{network_size}_Analysis{i}_modified.csv'
                analyses_modified.append(analysis)
                analyzer.saveAnalysis(analysis, filename)

            ave_time = sum(time_modified)/len(time_modified)
            print(f'Average time modified analysis took was {ave_time} seconds')
            print('Time for each modified analysis below:')
            print(time_modified)



    ## Compare full analyses vs. modified analyses
    for i in range(0, solutions_number):
        print('### ANALYSIS NO.' + str(i))
        analysis_full = analyses_full[i]
        analysis_modified = analyses_modified[i]
        for row_num, row_full in analysis_full.iterrows():
            row_modified = analysis_modified.loc[row_num]
            # motif_full = row_full['Motif']
            # motif_modified = row_modified['Motif']
            # if motif_full != motif_modified:
            #     print(f'Motif is not the same in row {row_num}')
            #     print('Full:')
            #     print(motif_full)
            #     print('modified')
            #     print(motif_modified)
            # num_full = row_full['Number of appearances in network']
            # num_modified = row_modified['Number of appearances in network']
            # if num_full != num_modified:
            #     print(f'Number of appearances is not the same in row {row_num}')
            #     print('modified is missing:')
            #     print(num_full-num_modified)
            indices_full = set(row_full['Edges indices'])
            indices_modified = set(row_modified['Edges indices'])
            if indices_full != indices_modified:
                print(f'Indices list is not the same in Motif No. {row_num}')
                if indices_full.difference(indices_modified):
                    print('Missing locations in modified analysis:')
                    print(indices_full.difference(indices_modified))
                if indices_modified.difference(indices_full):
                    print('Missing locations in full analysis:')
                    print(indices_modified.difference(indices_full))










