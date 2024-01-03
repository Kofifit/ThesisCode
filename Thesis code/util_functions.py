import pandas as pd
import numpy as np
import random
import copy
import itertools
import re

class UtilFunctions:

    @staticmethod
    def csv2network(filename):
        '''
        This function takes a file name as an input and return the content of the file as a dataframe of the network
        :param filename - string of the txt file name of the network:
        :return network - a numpy array that contains the network of the network:
        '''
        try:
            df = pd.read_csv(filename)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            return UtilFunctions.df2network(df)
        except Exception as error:
            return error


    @staticmethod
    def network2csv(filename, network, col_names):
        '''
        This function takes a file name as an input and return the content of the file as a dataframe of the network
        :param filename - string of the txt file name of the network:
        :return network - a numpy array that contains the network of the network:
        '''
        try:
            df = UtilFunctions.network2df(network, col_names)
            df.to_csv(filename, sep=',')
        except Exception as error:
            return error


    @staticmethod
    def df2network(df):
        df = df.astype(int)
        network = dict()
        for index, row in df.iterrows():
            network[row['Index']] = [[row['Gene A'], row['Gene B']], row['Activation/Repression'], row['delta']]
        return network


    @staticmethod
    def network2df(network, col_names):
        df_origin = pd.DataFrame.from_dict(network, orient='index', columns=col_names)
        df_genes = pd.DataFrame(df_origin.edge.tolist(), index=df_origin.index, columns=['Gene A', 'Gene B'])
        df_origin.drop(columns=['edge'], inplace=True)
        df = pd.concat([df_genes, df_origin], axis=1, join="inner")
        df.sort_index()
        return df


    @staticmethod
    def addMotifs2Network(network, motifs_df):
        for key in network.keys():
            network[key].extend(np.zeros(len(motifs_df), dtype=int))
        index = -len(motifs_df)
        for i, motif in motifs_df.iterrows():
            for key in motif['Edges indices']:
                network[key][index] = 1
            index += 1
        return network


    @staticmethod
    def csv2analysis(filename):
        '''
        This function takes a file name as an input and return the content of the file as a dataframe of the network
        :param filename - string of the txt file name of the analysis:
        :return network - a dataframe that contains the analysis of the network:
        '''
        try:
            df = pd.read_csv(filename)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            return UtilFunctions.analysis2df(df)
        except Exception as error:
            return error

    @staticmethod
    def analysis2df(analysis):
        new_indices = []
        indices = analysis['Edges indices']
        for i in indices:
            numbers = re.findall(r'\d+', i)
            new_indices.append([int(x) for x in numbers])
        analysis['Edges indices'] = new_indices
        return analysis


    @staticmethod
    def solutionSetFull2excel(solution_set, filename, col_names):
        '''
        This function gets a set of solutions and saves them in a txt file
        :param solution_set - a list of dataframes, each contains a solution for the network:
        :param filename - a file name in which the solution set would be saved:
        :return:
        '''
        with pd.ExcelWriter(filename) as writer:
            for index, solution in enumerate(solution_set):
                df = UtilFunctions.network2df(solution, col_names=col_names[index])
                sheet_name = '#' + str(index)
                df.to_excel(writer, sheet_name=sheet_name)


    @staticmethod
    def solutionSetModified2excel(solution_set, filename):
        '''
        This function gets a set of solutions and saves them in a txt file
        :param solution_set - a list of dataframes, each contains a solution for the network:
        :param filename - a file name in which the solution set would be saved:
        :return:
        '''
        with pd.ExcelWriter(filename) as writer:
            origin = solution_set[0]
            for index, solution in enumerate(solution_set):
                UtilFunctions.find_delta(origin, solution)
                df = UtilFunctions.network2df(solution)
                sheet_name = '#' + str(index)
                df.to_excel(writer, sheet_name=sheet_name)


    @staticmethod
    def excel2solutionSetList(filename):
        solutions = []
        df_dict = pd.read_excel(filename, sheet_name=None)
        for key in df_dict.keys():
            df = df_dict[key]
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            solution = UtilFunctions.df2network(df)
            solutions.append(solution)
        return solutions

    @staticmethod
    def edge2df(edge, df):
        index = edge[0]
        nodes = edge[1][0]
        func = edge[1][1]
        df.loc[len(df.index)] = [index, nodes[0], nodes[1], func]
        return df

    @staticmethod
    def generateRandSolutionSet(network, solution_number):
        '''
        This function gets a network and a desired number of solutions.
        Then it generate the solutions randomly and return them in a list of dataframes.
        :param solution_number:
        :param network - a dataframe of a network:
        :return solutions_set - a list of dataframes, each contains a solution:
        '''
        solutions_set = []
        while solution_number >= 0:
            solution_number -= 1
            edges_number = random.randint(int(len(network)*0.7), len(network))
            keys = random.sample(network.keys(), edges_number)
            solution_temp = {k: network[k] for k in keys if k in network}
            solutions_set.append(solution_temp)
        return solutions_set


    @staticmethod
    def find_delta(originNetwork, otherNetwork):

        originNetworkCopy = copy.deepcopy(originNetwork)

        for key in otherNetwork.keys():
            if key in originNetworkCopy:
                del originNetworkCopy[key]
                d = 0
            else:
                d = 1
            otherNetwork[key][-1] = d

        for key in originNetworkCopy.keys():
            value = originNetworkCopy[key]
            d = -1
            value[-1] = d
            otherNetwork[key] = value








