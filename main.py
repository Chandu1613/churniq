from src.data_loading_1 import data_loading
from src.data_cleaning_2 import data_cleaning
from src.data_preprocessing_3 import data_preprocessing
from src.logger import log_message
df = data_loading()

df = data_cleaning(df)

X_train,X_test,y_train,y_test = data_preprocessing(df)
