import numpy as np
import arviz as az

class ConversionRate():
    def __init__(self, name, a, b):
        self.name = name
        self.a = a
        self.b = b
    
    def __repr__(self):
        return f"Conversion Rate: '{self.name}' (alpha:{self.a}, beta:{self.b})"
    
    def update(self, conversion):
        self.a += conversion
        self.b += 1 - conversion
        
    def sample_mean(self):
        return np.random.beta(self.a, self.b, size=1)
    
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
        
    def update(self, occurrences):
        self.a += occurrences
        self.b += 1
        
    def sample_mean(self):
        return np.random.gamma(self.a, 1 / self.b, size=1)
    
    def hdi(self, hdi_prob):
        lower, upper = az.hdi(self.samples, hdi_prob)
        return lower, upper
    
class ThompsonSampling():
    def __init__(self):
        self.bandits = []
        self.bandits_primed = []
        self.curr_bandit = None
        
    def add_bandit(self, new_bandit):
        for bandit in self.bandits:
            if bandit.name == new_bandit.name:
                Exception(f"Bandit with name '{new_bandit.name}' already exists")
                break
                
        self.bandits.append(new_bandit)
        self.bandits_primed.append(False)
        
    def _prime(self):
        bandit_idx = np.random.choice(np.where(np.array(self.bandits_primed) == False)[0])
        self.bandits_primed[bandit_idx] = True
        return self.bandits[bandit_idx], bandit_idx
    
    def get_bandit(self):
        if not np.all(self.bandits_primed):
            return self._prime()
        samples = []
        for bandit in self.bandits:
            samples.append(bandit.sample_mean())
        bandit_choice = np.argmax(samples)
        return self.bandits[bandit_choice], bandit_choice
    
    def update(self, bandit_used, outcome):
        if isinstance(bandit_used, (str, int)):
            bandit_used_name = bandit_used
        else:
            bandit_used_name = bandit_used.name
            
        for bandit in self.bandits:
            if bandit.name == bandit_used_name:
                bandit.update(outcome)
                return
        Exception(f"Bandit with name '{bandit_used_name}' not found")