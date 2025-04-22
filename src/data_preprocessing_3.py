import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from src.logger import log_message



def data_preprocessing(df):
    numerical = []
    categorical = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical.append(col)
        else:
            numerical.append(col)
    log_message("Checking for duplicates...")
    if df.duplicated().sum() > 0:
        log_message(f"DataFrame has {df.duplicated().sum()} duplicate rows")
        df = df.drop_duplicates()
        log_message(f"DataFrame has {df.duplicated().sum()} duplicate rows after dropping duplicates")

    log_message('Checking for missing values')
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                log_message(f"Column {col} has {df[col].isnull().sum()} missing values")
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                log_message(f"Column {col} has {df[col].isnull().sum()} missing values")
                df.loc[:, col] = df[col].fillna(round(df[col].mean()))
                log_message(f"Column {col} has been filled with mean value: {round(df[col].mean())}")
    log_message('Checking for Outliers...')
    df_copy = df.copy()
    def outliers(df):
        for col in df.columns:
            if col in numerical[:-1]:
                if col != 'Age':
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    log_message(f"Column {col} has {len(outliers)} outliers")
                    if outliers.shape[0] > 0:
                        df[col] = df[col].clip(lower_bound, upper_bound).astype(int)
                        log_message(f"Outliers in column {col} have been clipped to the lower and upper bounds")
        return df

    df_cleaned = outliers(df_copy)
    
    log_message('Splitting the data into train and test sets...')
    X, y = df_cleaned.drop(columns=['Exited']), df_cleaned['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_message('Encoding categorical variables...')
    X_train_dummies = pd.get_dummies(X_train, columns=categorical, drop_first=True)
    X_test_dummies = pd.get_dummies(X_test, columns=categorical, drop_first=True)

    X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)
    
    log_message('Saving the encoded columns...')
    joblib.dump(X_train_dummies.columns.tolist(), '/home/user/churniq/artifacts/encoded_columns.pkl')

    log_message('Scaling the data...')
    X_train_std = X_train_dummies.copy()
    X_test_std = X_test_dummies.copy()

    std = StandardScaler()

    X_train_std = std.fit_transform(X_train_std)
    X_test_std = std.transform(X_test_std)

    log_message('Saving the scaler...')
    joblib.dump(std, '/home/user/churniq/artifacts/scaler_std.pkl')
    
    log_message("Checking for Imbalance data....")
    if (df['Exited'].value_counts()*100/df.shape[0])[1] != (df['Exited'].value_counts()*100/df.shape[0])[0]:
        log_message('HAndling the imbalance data....')
        smote = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_std, y_train)
    
    return X_train_resampled, X_test_std, y_train_resampled, y_test