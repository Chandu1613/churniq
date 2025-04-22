from src.data_loading_1 import data_loading
from src.data_cleaning_2 import data_cleaning 

df = data_loading()

df = data_cleaning(df)

print(df.head())