from functools import partial

from modelling.base import get_code_channel_pairs
from modelling.db import DB_NAME
from modelling.functions import experiment
from polar_codes import FastSSCPolarCode

COLLECTION = 'fast_ssc'
CODE_RATES = [0.25, 0.33, 0.5, 0.66, 0.75, ]
SNR_RANGE = [i/2 for i in range(2, 9)]
MESSAGES_PER_EXPERIMENT = 1000
REPETITIONS = 50


get_pairs = partial(
    get_code_channel_pairs,
    code_cls=FastSSCPolarCode,
    rates=CODE_RATES,
    snr_range=SNR_RANGE,
    repetitions=REPETITIONS,
)


def fast_ssc_experiment(args):
    print(f'({args[0].N}, {args[0].K}) Code, SNR (dB) = {args[1].snr_db}')

    func = partial(
        experiment,
        db_name=DB_NAME,
        collection=COLLECTION,
        messages=MESSAGES_PER_EXPERIMENT,
    )
    return func(*args)
