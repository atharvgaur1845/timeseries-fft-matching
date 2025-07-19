import pandas as pd
input_csv_path = 'data.csv'
train_csv_path = 'train_data_70.csv'
test_csv_path = 'test_data_30.csv'

split_ratio = 0.7
df = pd.read_csv(input_csv_path)
#df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(df) * split_ratio)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]
df_train.to_csv(train_csv_path, index=False)
df_test.to_csv(test_csv_path, index=False)

    

