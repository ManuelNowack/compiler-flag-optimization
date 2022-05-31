class Tuner:
    def __init__(self, search_space, evaluator, name, default_setting):
        self.search_space = search_space
        self.evaluator = evaluator
        self.name = name
        self.default_setting = default_setting
        self.default_perf = evaluator.evaluate(default_setting)

    
    def tune(self, budget, batch_size=1, file=None):
        raise NotImplementedError
