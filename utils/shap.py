import pandas as pd
import shap

def tree_model_shap_table(model, X, num_features_to_show: int=50):
    """Find average SHAP value for each feature.

    Args:
        model (_type_): _description_
        X (pd.DataFrame): dataframe used to train the model, containing column names
    """
    shap_explainer = shap.TreeExplainer(model)
    
    # Obtain shap values for each observation for each feature
    shap_values = shap_explainer.shap_values(X, check_additivity=False)
    
    # aggregate SHAP values
    df = pd.DataFrame(shap_values, columns=X.columns)
    df = df.abs().mean()
    df = (df / df.sum()).reset_index().fillna(0)
    shap_agg = df.rename(columns={'index': 'Features', 0: 'Average SHAP Value (%)'})
    
    shap_agg_drop = shap_agg.set_index('Features') \
        .sum(axis=1).sort_values().reset_index().tail(num_features_to_show)[['Features']]
    shap_agg = shap_agg_drop.merge(shap_agg, on="Features", how="left")
    
    return shap_agg