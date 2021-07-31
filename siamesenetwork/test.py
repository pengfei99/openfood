import sys
from utils import parse_input_argv


def main(argv):
    cfile, lrate, nepochs = parse_input_argv(argv)
    print(cfile)
    print(lrate)
    print(nepochs)
    conf_file = cfile or "pengfei.txt"
    learning_rate = lrate or 0.888
    n_epochs = nepochs or 50
    print(conf_file)
    print(learning_rate)
    print(n_epochs)


if __name__ == "__main__":
    main(sys.argv[1:])
