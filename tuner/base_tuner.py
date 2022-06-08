class Tuner:
    def __init__(self, search_space, evaluator, name, default_optimization):
        self.search_space = search_space
        self.evaluator = evaluator
        self.name = name
        self.default_optimization = default_optimization
        self.default_perf = evaluator.evaluate(default_optimization, 10)

    
    def tune(self, budget, batch_size=1, file=None):
        raise NotImplementedError
