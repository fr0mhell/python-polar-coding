import multiprocessing
from concurrent import futures
from typing import Dict

from ..channels import SimpleAWGNChannel
from ..modems import SimpleBPSKModem
from ..polar_codes import FastSSCPolarCode, RCSCANPolarCode
from . import functions
from .http import get_params, save_result


class CodeTypes:
    """Code types"""
    FAST_SSC = 'fast-ssc'
    RC_SCAN = 'rc_scan'
    ALL = [FAST_SSC, RC_SCAN]


class ChannelTypes:
    SIMPLE_BPSK = 'simple-bpsk'


CODE_MAP = {
    CodeTypes.FAST_SSC: FastSSCPolarCode,
    CodeTypes.RC_SCAN: RCSCANPolarCode,
}


MODEM_MAP = {
    ChannelTypes.SIMPLE_BPSK: SimpleBPSKModem,
}


def simulate(code_type: str, channel_type: str, snr: float, messages: int,
             code_params: Dict) -> Dict:
    """Simulate polar codes transmission."""
    code = CODE_MAP[code_type](**code_params)
    modem = MODEM_MAP[channel_type](fec_rate=code.K/code.N, snr_db=snr)
    channel = SimpleAWGNChannel()

    bit_errors, frame_errors = 0, 0

    for _ in range(messages):
        message = functions.generate_binary_message(code.K)
        encoded = code.encode(message)
        modulated = modem.modulate(encoded)
        transmitted = channel.transmit(modulated)
        demodulated = modem.demodulate(transmitted)
        decoded = code.decode(demodulated)

        be, fe = functions.compute_fails(expected=message, decoded=decoded)
        bit_errors += be
        frame_errors += fe

    return {
        'snr_db': snr,
        'bits': messages * code.K,
        'bit_errors': bit_errors,
        'frames': messages,
        'frame_errors': frame_errors,
    }


def simulate_from_params(experiment: Dict, url: str):
    """Simulate polar code chain using remote params."""
    channel_type = experiment.pop('channel_type')
    code_id = experiment.pop('code_id')
    code_type = experiment.pop('code_type')
    snr = experiment.pop('snr')
    messages = experiment.pop('messages')
    # Pop `type` to prevent problems with initialization
    typ = experiment.pop('type')

    result = simulate(
        code_type=code_type,
        channel_type=channel_type,
        snr=snr,
        messages=messages,
        code_params=experiment,
    )

    print(f'Result: {result}\n'
          f'{code_type.upper()} ({experiment["N"]}, {experiment["K"]})')

    save_result(
        url=url,
        result=result,
        code_id=code_id,
        code_type=code_type,
        channel_type=channel_type
    )


def simulate_multi_core(experiments: int, url: str):
    """Simulate polar code chain using multiple cores."""
    params_for_experiments = get_params(url=url, experiments=experiments)
    workers = multiprocessing.cpu_count()

    print(f'Workers: {workers}\n'
          f'Number of experiments: {len(params_for_experiments)}')

    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        run_tasks = {
            ex.submit(simulate_from_params, *(params, url)): (params, url)
            for params in params_for_experiments
        }
        for future in futures.as_completed(run_tasks):
            try:
                future.result()
            except Exception as exc:
                print(exc)
