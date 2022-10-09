"""
The main file to run experiments.
"""

import argparse


def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    return args


def main():
    args = options()

    # do something ...


if __name__ == '__main__':
    main()
