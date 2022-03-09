FLOAT_MAX = float('inf')
FLOAT_MIN = -float('inf')


class MinMaxStats:
    def __init__(self, ):
        self.minimum = float('inf')
        self.maximum = - float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum,value)
        self.minimum = min(self.minimum,value)

    def normalize(self, value: float):
        norm_value = value
        if self.maximum > self.minimum:
            norm_value = (norm_value - self.minimum) / (self.maximum - self.minimum)
        return norm_value


class MinMaxStatsList:
    def __init__(self, num):
        self.num = num
        self.stats_lst = [MinMaxStats() for _ in range(self.num)]
