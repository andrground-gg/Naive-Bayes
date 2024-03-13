def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision(y_true, y_pred):
    return (y_true & y_pred).sum() / y_pred.sum()

def recall(y_true, y_pred):
    pass

def f1(y_true, y_pred):
    pass