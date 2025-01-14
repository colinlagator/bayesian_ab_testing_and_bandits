import numpy as np
import arviz as az

class ConversionRate():
    def __init__(self, name, a, b):
        self.name = name
        self.a = a
        self.b = b
        self.samples = None
    
    def __repr__(self):
        return f"Conversion Rate: '{self.name}' (alpha:{self.a}, beta:{self.b})"
    
    def update(self, num_conversions, num_samples):
        self.a += num_conversions
        self.b += num_samples - num_conversions
        self.samples = self._sample(100000)
        
    def _sample(self, num_samples):
        samples = np.random.beta(self.a, self.b, size=num_samples)
        return samples
    
    def hdi(self, hdi_prob):
        lower, upper = az.hdi(self.samples, hdi_prob)
        return lower, upper
    
class Demand():
    def __init__(self, name, a, b):
        self.name = name
        self.a = a
        self.b = b
        self.samples = None
        
    def __repr__(self):
        return f"Demand: '{self.name}' (alpha:{self.a}, beta:{self.b})"
        
    def update(self, occurrences, intervals):
        self.a += occurrences
        self.b += intervals
        self.samples = self._sample(100000)
        
    def _sample(self, num_samples):
        samples = np.random.gamma(self.a, 1 / self.b, size=num_samples)
        return samples
    
    def hdi(self, hdi_prob):
        lower, upper = az.hdi(self.samples, hdi_prob)
        return lower, upper
    
def a_b_test(control, treatment, eps, diff=0):
    """
    Parameters:
    control - the control
    treatment - the treatment
    diff - the minimum amount that treatment must be greater than control by, 
        default is 0, which tests if treatment is strictly greater than control
    eps - the maximum expected loss that treatment can have compared to control
    """
    expected_loss = np.maximum(control.samples - treatment.samples, diff).mean()
    expected_loss_less_than_eps = expected_loss < eps
    if not expected_loss_less_than_eps:
        return control
    return treatment

def multivariate_test(*treatments, control, eps, diff=0):
    """
    Parameters:
    control - the control
    treatment - the treatment
    diff - the minimum amount that treatment must be greater than control by, 
        default is 0, which tests if treatment is strictly greater than control
    eps - the maximum expected loss that treatment can have compared to control
    """
    treatments_samples = np.vstack([a.samples for a in treatments])
    expected_losses = np.maximum(control.samples - treatments_samples, diff).mean(axis=1)
    expected_loss_less_than_eps = expected_losses < eps
    if not np.any(expected_loss_less_than_eps):
        return control
    
    prob_greater_than_all = []
    for i in range(treatments_samples.shape[0]):
        mask = np.array([i for i in range(treatments_samples.shape[0])]) == i
        prob_greater_than_all.append((treatments_samples[mask] > np.max(treatments_samples[~mask], axis=0)).mean(axis=1)[0])
    prob_greater_than_all = np.array(prob_greater_than_all)
    
    is_best_treatment = np.array([i for i in range(len(prob_greater_than_all))]) == prob_greater_than_all.argmax()
    is_best = is_best_treatment & expected_loss_less_than_eps
    
    return np.array(treatments)[is_best][0]