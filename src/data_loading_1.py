import pandas as pd
from src.logger import log_message

def data_loading():
    """This function loads the messy data and gives a single dataset...
    """

    log_message('Data loading.....')
    dict_df = pd.read_excel(r'/home/user/churniq/data/Bank_Churn_Messy.xlsx',sheet_name=['Customer_Info','Account_Info'])
    customer_df = dict_df.get('Customer_Info')
    acc_df = dict_df.get('Account_Info')
    df = pd.merge(customer_df, acc_df, how='inner', on=['CustomerId','Tenure'])
    log_message("Data loaded successfully.....")
    
    return df