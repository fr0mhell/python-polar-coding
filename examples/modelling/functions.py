from math import ceil
from random import shuffle
from datetime import datetime

import numpy as np
from pymongo import MongoClient

from examples.modelling.mongo import URI


def single_transmission(code, channel):
    """Transmit a message through a simple BPSK model.

    Args:
        code (PolarCode): Polar code tp simulate.
        channel (PSKModAWGNChannel): Transmission channels with configured SNR.

    Returns:
        bit_errors (int): Number of bit errors occurred while message
            transmission.
        word_errors (int): Number of messages transmitted with errors.

    """
    message = np.random.randint(0, 2, code.K)
    encoded = code.encode(message)
    llr = channel.transmit(encoded)
    decoded = code.decode(llr)

    fails = np.sum(message != decoded)
    bit_errors = int(fails)
    word_errors = int(fails > 0)

    return bit_errors, word_errors


def simulation_task(code, channel, db_name, collection, messages=1000):
    start = datetime.now()

    client = MongoClient(URI)
    bit_errors = word_errors = 0

    for m in range(messages):
        be, we = single_transmission(code, channel)
        bit_errors += be
        word_errors += we

    data = code.to_dict()

    end = datetime.now()
    data.update({
        'snr_db': channel.snr_db,
        'bits': messages * code.K,
        'bit_errors': bit_errors,
        'word_errors': word_errors,
        'words': messages,
        'channels': str(channel),
        'start': start,
        'end': end
    })

    client[db_name][collection].insert_one(data)

    iterations = getattr(code, '_iterations', -1)
    print(f'Execution took {end - start} ({iterations}).\n')


def generate_simulation_parameters(
    code_cls,
    channel_cls,
    N,
    code_rates,
    snr_range,
    repetitions,
    additional_code_params=None
):
    """Get list of (PolarCode, Channel) pairs."""
    additional_code_params = additional_code_params or [{}, ]

    combinations = [
        (
            code_cls(
                codeword_length=N,
                info_length=ceil(N * cr),
                is_systematic=True,
                **ap,
            ),
            channel_cls(
                snr_db=snr,
                N=N,
                K=ceil(N * cr)
            )
        ) for cr in code_rates for snr in snr_range
          for ap in additional_code_params
    ] * repetitions

    shuffle(combinations)

    return combinations
