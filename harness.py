class modelFailure(Exception):
    """Trigger when model needs to evolve"""


class evoHarness:
    def __init__(self):
        self.needToEvolve = False

    def run(self):
        try:
            #MAIN MODEL HERE
            raise modelFailure()
        except modelFailure:
            self.needToEvolve = True

            return False

    def evolve(self):
        if self.needToEvolve:
            pass

    def benchmark():
        pass

    def regress():
        pass
