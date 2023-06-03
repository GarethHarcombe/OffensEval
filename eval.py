from sklearn.metrics import confusion_matrix, f1_score

def f1(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))


def print_results(test_labels, preds):
    """
    print_results: print results from predicted labels vs gold standard labels
    
    Inputs:
        test_labels: list(any), gold standard labels to test against
        preds: list(any), labels predicted by the model
    """
    results = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = results.ravel()
    print("Accuracy: {:.4f}".format((tn + tp) / len(preds)))
    print("Confusion matrix: \n", results)
    print("F1: {:.4f}".format(f1_score(test_labels, preds, average="macro")))
