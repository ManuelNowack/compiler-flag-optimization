# Define constant
FLOAT_MAX = float('inf')

class Tuner:
    def __init__(self, search_space, evaluator, name = "Base Tuner", default_setting = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.name = name
        self.default_setting = default_setting
        self.default_perf = evaluator.evaluate(default_setting)
        self.visited = set()

        print(f"default_perf : {self.default_perf:.3f}")

    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"

    def tune(self, budget, batch_size=1, file=None):
        best_opt_setting, best_perf = None, FLOAT_MAX
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
        return best_opt_setting, best_perf