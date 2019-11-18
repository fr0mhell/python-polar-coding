from concurrent.futures import ProcessPoolExecutor
from math import ceil
from random import shuffle
from datetime import datetime

from modelling.channel import VerificationChannel


def get_code_channel_pairs(code_cls, N, rates, snr_range, repetitions):
    """Get list of (PolarCode, Channel) pairs."""
    combinations = [
        (
            code_cls(
                codeword_length=N,
                info_length=ceil(N * cr),
                is_systematic=True),
            VerificationChannel(snr_db=snr, N=N, K=ceil(N * cr))
        ) for cr in rates for snr in snr_range
    ] * repetitions

    shuffle(combinations)

    return combinations


def run_model(workers, function, param_list):
    """"""
    print('Start execution at', datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        ex.map(function, param_list)

    print('Finish execution at', datetime.now().strftime('%H:%M:%S %d-%m-%Y'))
