import argparse
from avatar.selection import *
from avatar.analysis import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment")
    parser.add_argument("-c", "--chunk", type=str, default="1/1")
    parser.add_argument("-s", "--selector", type=str, default=None)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    args = parser.parse_args()

    if args.selector:
        selector = eval(args.selector)
        print(selector)