from util_functions import UtilFunctions
from main import runAnalysis
import itertools
import pandas as pd

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
    # UtilFunctions.network2csv('temp_network.csv', new_network, col_names)
    n = 3
    filename = 'Solutions/solutionsModified.xlsx'
    new_filename = 'Solutions/solutionsModifiedWithMotifs.xlsx'
    solutions = UtilFunctions.excel2solutionSetList(filename)
    all_col_names = []
    solution_set = []
    for i, s in enumerate(solutions):
        if len(s) > 50:
            s = dict(itertools.islice(s.items(), 50))
        filename = 'Analysis' + str(i) + '.csv'
        motifs_df = runAnalysis(n, s, filename)
        new_network = UtilFunctions.addMotifs2Network(s, motifs_df)
        col_names = ['edge', 'Activation/Repression', 'delta']
        for motif_number, row in motifs_df.iterrows():
            name = 'Motif #' + str(motif_number)
            col_names.append(name)
        all_col_names.append(col_names)
        solution_set.append(new_network)
    UtilFunctions.solutionSetFull2excel(solution_set, new_filename, all_col_names)





