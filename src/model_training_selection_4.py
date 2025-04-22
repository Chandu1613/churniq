from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.logger import log_message
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve, auc

def model_train_selection(X_train, X_test, y_train, y_test):
    log_message("Models initiating....")
    models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "LightGBM":LGBMClassifier(random_state=42)
    }

    log_message('Model training and testing.....')
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "precision_class_1": report["1"]["precision"],
            "recall_class_1": report["1"]["recall"],
            "f1_score_class_1": report["1"]["f1-score"],
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Model'}, inplace=True)

    results_df = results_df.sort_values(by='f1_score_class_1', ascending=False).reset_index(drop=True)
    log_message('Selecting the top 2 models...')

    return models[results_df.iloc[0, 0]],models[results_df.iloc[1, 0]]