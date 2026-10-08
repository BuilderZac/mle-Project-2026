import harness


def __main__():
    model = harness.evoHarness()

    while not model.run():
        model.evolve()
        while not model.benchmark():
            model.regress()
            model.evolve()
