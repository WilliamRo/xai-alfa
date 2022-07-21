import pandas as pd



file_path = r'E:\xai-alfa\51-SLEEP\data\ucddb\SubjectDetails.xls'

df: pd.DataFrame = pd.read_excel(file_path)

detail = df.loc[df['Study Number'] == 'UCDDB002'].to_dict(
  orient='index').popitem()[1]








print(df)


