import pandas as pd


def converter(filename, new_filename):

    # Get network from csv file in dataframe
    df = pd.read_csv(filename)

    # Get names of all genes and number them in a dictionary
    gene_dict = {}
    function_dict = {}
    counter_g = 1
    counter_f = 1
    for index, row in df.iterrows():
        for n in range(0, 2):
            name = row.iloc[n]
            if name not in gene_dict.keys():
                gene_dict[name] = counter_g
                counter_g += 1
        function = row.iloc[2]
        if function not in function_dict.keys():
            function_dict[function] = counter_f
            counter_f += 1

    # Replace gene names to numbers in df
    for index, row in df.iterrows():
        for n in range(0, 2):
            df.iat[index, n] = gene_dict[df.iat[index, n]]
        df.iat[index, 2] = function_dict[df.iat[index, 2]]

    # Save changes into new csv file
    df.to_csv(new_filename, index=False)


def main():
    filename = input('Please enter name of network csv file ')
    new_filename = input('Please enter name for new file for numbered network ')
    converter(filename.strip(), new_filename.strip())

if __name__ == '__main__':
    main()


