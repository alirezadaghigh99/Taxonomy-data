import numpy as np
from scipy.special import logit, expit

def get_label_quality_ensemble_scores(labels, pred_probs_list, method='log_loss', adjust_pred_probs=False, 
                                      weight_ensemble_members_by='uniform', custom_weights=None, 
                                      log_loss_search_T_values=None, verbose=False):
    def log_loss(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def adjust_probs(probs):
        return expit(logit(probs) * 1.1)  