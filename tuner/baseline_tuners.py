from .common import Tuner, FLOAT_MAX
import random, time

class RandomTuner(Tuner):
    def __init__(self, search_space, evaluator, default_setting):
        super().__init__(search_space, evaluator, "Random Tuner", default_setting)
        
        
    def generate_candidates(self, batch_size=1):
        random.seed(time.time())
        candidates = []
        for _ in range(batch_size):
            while True:
                opt_setting = dict()
                for flag_name, flag_info in self.search_space.items():
                    num = len(flag_info.configs)
                    rv = random.randint(0, num-1)
                    opt_setting[flag_name] = flag_info.configs[rv]
                
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

class mPBILTuner(Tuner):
    def __init__(self, search_space, evaluator, learning_rate=1):
        super().__init__(search_space, evaluator, "mPBIL Tuner")
        self.learning_rate = learning_rate
 
class OpenTuner(Tuner):
    def __init__(self, search_space, evaluator):
        super().__init__(search_space, evaluator)






