import numpy as np
import pandas as pd

def evaluate_thresholds(y_true, probabilities, thresholds, false_positive_cost=50, false_negative_cost=1,):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths")

    if len(y_true) == 0:
        raise ValueError("There is no data for evaluation")
        
    results = []


    
    for threshold in thresholds:


        predictions = (probabilities >= threshold).astype(int)

        tp = ((y_true == 1) & (predictions == 1)).sum()
        tn = ((y_true == 0) & (predictions == 0)).sum()
        fp = ((y_true == 0) & (predictions == 1)).sum()
        fn = ((y_true == 1) & (predictions == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if ((precision + recall) > 0) else 0
        
        predicted_positive_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0 
        
        business_cost = false_positive_cost * fp + false_negative_cost * fn 
        
        results.append({"threshold": threshold, 
                        "tp": tp, 
                        "fp": fp, 
                        "tn": tn, 
                        "fn": fn, 
                        "precision": precision, 
                        "recall": recall, 
                        "false_positive_rate": false_positive_rate, 
                        "specificity": specificity, 
                        "f1": f1, 
                        "predicted_positive_rate": predicted_positive_rate, 
                        "business_cost": business_cost
                       })
                       
    return pd.DataFrame(results)