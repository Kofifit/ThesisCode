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

    ## Generate random solutions
    filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/network.csv'
    network = UtilFunctions.csv2network(filename)
    network = dict(itertools.islice(network.items(), 30))
    solution_set = UtilFunctions.generateRandSolutionSet(network, 10)
    filename_full = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsFull_test.xlsx'
    filename_modified = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsModified_test.xlsx'
    col_names = ['edge', 'Activation/Repression', 'delta']
    UtilFunctions.solutionSetFull2excel(solution_set, col_names, filename_full)
    UtilFunctions.solutionSetModified2excel(solution_set, col_names, filename_modified)

    ## Test delta extraction on full solutions
    n = 3
    filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsFull_test.xlsx'
    solutions = UtilFunctions.excel2solutionSetList(filename)
    analyses_full = []
    for i, s in enumerate(solutions):
        start_time = time()
        analyzer = DeltaNetworkMotifAnalyzer(s, n)
        analysis = analyzer.originAnalysis
        analyses_full.append(analysis)
        filename = 'AnalysisFullTest' + str(i) + '.csv'
        analyzer.saveAnalysis(analyzer.originAnalysis, filename)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f'#{i} Analysis for full solution took: {elapsed_time} seconds')


    ## Test delta extraction on modified solutions
    n = 3
    filename = '/Users/renanabenyehuda/PycharmProjects/ThesisAlgorithm/Thesis code/Solutions/solutionsModified_test.xlsx'
    solutions = UtilFunctions.excel2solutionSetList(filename)
    analyses_modified = []
    for i, s in enumerate(solutions):
        start_time = time()
        if i == 0:
            analyzer = DeltaNetworkMotifAnalyzer(s, n)
            analysis = analyzer.originAnalysis
        else:
            extractor = NetworkDeltaExtractor(n, s)
            extractor.extractDeltaNetwork()
            delta = extractor.getDeltaNetwork()
            deltaNetwork = NetworkDisessembler(delta).getNetwork()
            analysis = analyzer.analyze(deltaNetwork)
            analysis = analyzer.compare(s, analysis)
        filename = 'AnalysisModifiedTest' + str(i) + '.csv'
        analyses_modified.append(analysis)
        analyzer.saveAnalysis(analysis, filename)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f'#{i} Analysis modified solution took: {elapsed_time} seconds')


    ## Compare full analyses vs. modified analyses
    for i in range(0, 10):
        analysis_full = analyses_full[i]
        analysis_modified = analyses_modified[i]
        for row_num, row_full in analysis_full.iterrows():
            row_modified = analysis_modified.loc[row_num]
            motif_full = row_full['Motif']
            motif_modified = row_modified['Motif']
            if motif_full != motif_modified:
                print(f'Motif is not the same in row {row_num}')
            num_full = row_full['Number of appearances in network']
            num_modified = row_modified['Number of appearances in network']
            if num_full != num_modified:
                print(f'Number of appearances is not the same in row {row_num}')
            indices_full = set(row_full['Edges indices'])
            indices_modified = set(row_modified['Edges indices'])
            if indices_full != indices_modified:
                print(f'Indices list is not the same in row {row_num}')
            locations_full = set(row_full['Location of appearances in network'])
            locations_modified = set(row_modified['Location of appearances in network'])
            if locations_full != locations_modified:
                print(f'Locations list is not the same in row {row_num}')









