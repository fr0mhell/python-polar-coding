import numpy as np

from modelling.db import client


def single_transmission(code, channel):
    """Transmit a message through a simple BPSK model.

    Args:
        code (PolarCode): Polar code tp simulate.
        channel (PSKModAWGNChannel): Transmission channel with configured SNR.

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


def experiment(code, channel, db_name, collection, messages=1000):
    bit_errors = word_errors = 0

    for m in range(messages):
        be, we = single_transmission(code, channel)
        bit_errors += be
        word_errors += we

    data = code.to_dict()
    data.update({
        'snr_db': channel.snr_db,
        'bits': messages * code.K,
        'bit_errors': bit_errors,
        'word_errors': word_errors,
        'channel': str(channel),
    })

    client[db_name][collection].insert_one(data)
