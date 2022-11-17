import game.tests as tests
import game.algorithms as alg


def main():
    gs = tests.GridSearch()
    # gs.launcher()
    gs.analysis()
    gs.plot_analysis()

    return 1


if __name__ == "__main__":
    main()
