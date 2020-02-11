import numpy as np
from numba import njit


@njit
def generate_binary_message(size):
    """Generate binary message."""
    return np.random.randint(0, 2, size)


@njit
def compute_fails(expected, decoded):
    """Compute number of fails in decoded message comparing with expected."""
    bit_errors = np.sum(expected != decoded)
    frame_error = int(bit_errors > 0)
    return bit_errors, frame_error


def transmission(code, channel, snr_db):
    message = np.random.randint(0, 2, code.K)
    encoded = code.encode(message)
    llr = channel.transmit(message=encoded, snr_db=snr_db)
    decoded = code.decode(llr)

    fails = compute_fails(expected=message, decoded=decoded)
    bit_errors = fails
    frame_errors = int(fails > 0)

    return bit_errors, frame_errors


def simulation_task(code, channel, snr_db, messages):
    bit_errors, frame_errors = 0, 0

    for m in range(messages):
        be, fe = transmission(code, channel, snr_db)
        bit_errors += be
        frame_errors += fe

    return {
        'snr_db': snr_db,
        'bits': messages * code.K,
        'bit_errors': bit_errors,
        'frames': messages,
        'frame_errors': frame_errors,
    }
