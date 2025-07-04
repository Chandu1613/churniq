import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.logger import log_message

def data_cleaning(df):

    try:
        log_message('Dropping columns...')
        df = df.drop(columns=['CustomerId','Surname'],axis=1)
    except:
        log_message('Columns already dropped')
    
    df['EstimatedSalary'] = df['EstimatedSalary'].astype(str).str.replace('€', '', regex=False)
    df['Balance'] = df['Balance'].astype(str).str.replace('€', '', regex=False)

    df['EstimatedSalary'] = df['EstimatedSalary'].astype(float)
    df['Balance'] = df['Balance'].astype(float)

    df['EstimatedSalary'] = df['EstimatedSalary'].apply(lambda x:None if x== -999999.00 else x)

    if 'French' or 'FRA' in df['Geography']:
        df['Geography'] = df['Geography'].apply(lambda x:'France' if x=='French' else x)
        df['Geography'] = df['Geography'].apply(lambda x:'France' if x=='FRA' else x)
    log_message('Data is cleaned...')
    
    return df