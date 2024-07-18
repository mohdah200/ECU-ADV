import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    # Load ground truth labels
    ground_truth = pd.read_csv(test_annotation_file)
    
    # Load user predictions
    user_predictions = pd.read_csv(user_annotation_file)
    
    # Ensure the columns match
    y_true = ground_truth['class3']
    y_pred = user_predictions['class3']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Prepare the results dictionary
    results = {
        "result": [
            {
                phase_codename: {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                }
            }
        ]
    }

    return results

if __name__ == "__main__":
    import json
    import sys

    # The arguments will be passed by EvalAI system
    test_annotation_file = sys.argv[1]
    user_annotation_file = sys.argv[2]
    phase_codename = sys.argv[3]

    # Evaluate the results
    results = evaluate(test_annotation_file, user_annotation_file, phase_codename)

    # Print the results in JSON format
    print(json.dumps(results))
