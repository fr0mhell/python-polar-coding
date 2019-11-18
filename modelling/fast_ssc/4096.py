from modelling.base import run_model
from modelling.fast_ssc.params import fast_ssc_experiment, get_pairs

CODE_LENGTH = 4096
MAX_WORKERS = 8

code_channel_pairs = get_pairs(N=CODE_LENGTH)


if __name__ == '__main__':
    run_model(MAX_WORKERS, fast_ssc_experiment, code_channel_pairs)
