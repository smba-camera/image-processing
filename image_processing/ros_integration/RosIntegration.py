
class RosIntegration:
    def __init__(self):
        self.a = 1

    def registerRosFunction(self, name, callback):
        # resgister function on ros master
        name=name