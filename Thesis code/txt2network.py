import pandas as pd

def txt2dataframe(filename):
    try:
        df = pd.read_csv(filename, sep=" ", header=None)
        df.columns = ['Gene A', 'Gene B', 'Activation/Repression']
        df = df.astype(int)
        return df
    except Exception as error:
        return error
