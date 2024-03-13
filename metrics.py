def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision(y_true, y_pred):
    return (y_true & y_pred).sum() / ((y_true & y_pred).sum() + (~y_true & y_pred).sum())

def recall(y_true, y_pred):
    return (y_true & y_pred).sum() / ((y_true & y_pred).sum() + (y_true & ~y_pred).sum())

def f1(y_true, y_pred):
    return 2 * precision(y_true, y_pred) * recall(y_true, y_pred) / (precision(y_true, y_pred) + recall(y_true, y_pred))