import pandas as pd


df = pd.DataFrame(
    dict(A=range(1,4),B=range(4,7),C=range(7,10)),
    columns = list("ABC"),
    index = list('xyz'),
)
print(df)
method = 'records'
df.to_json(f'df_{method}.json',orient=f'{method}',lines=True)