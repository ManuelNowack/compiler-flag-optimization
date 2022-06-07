from .base_tuner import Tuner
import random, time

class RandomTuner(Tuner):
    def __init__(self, search_space, evaluator, default_setting):
        super().__init__(search_space, evaluator, "RandomTuner", default_setting)
        self.visited = set()
        
        
    def generate_candidates(self, batch_size=1):
        random.seed(time.time())
        candidates = []
        for _ in range(batch_size):
            while True:
                opt_setting = dict()
                for flag_name, configs in self.search_space.items():
                    num = len(configs)
                    rv = random.randint(0, num-1)
                    opt_setting[flag_name] = configs[rv]
                
                # Avoid duplication
                if str(opt_setting) not in self.visited:
                    self.visited.add(str(opt_setting))
                    candidates.append(opt_setting)
                    break
                
        return candidates
    
    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting) for opt_setting in candidates]

    def reflect_feedback(self, perfs):
        # Random search. Do nothing
        pass

    def tune(self, budget, batch_size=1, file=None):
        best_opt_setting, best_perf = None, float("inf")
        i = 0
        while i<budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
        
            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            print(best_perf, file=file)
            self.reflect_feedback(perfs)
        best_perf = self.evaluator.evaluate(best_opt_setting, 10)
        return best_opt_setting, best_perf
