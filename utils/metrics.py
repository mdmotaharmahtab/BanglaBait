from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import OrderedDict

def compute_metrics(pred):
    preds,labels = pred
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    # print(acc,precision,recall,f1)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    