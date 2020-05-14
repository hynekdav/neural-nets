class Architecture(object):
    def __init__(self, num_of_boxes):
        self.num_of_boxes = num_of_boxes

    def build(self, input_shape, optimizer, activation):
        raise NotImplementedError
