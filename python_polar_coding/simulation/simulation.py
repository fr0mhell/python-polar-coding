import multiprocessing
from typing import Dict
from concurrent import futures

from ..channels import SimpleAWGNChannel
from ..modems import SimpleBPSKModem
from ..polar_codes import (
    FastSSCPolarCodec,
    GFastSSCPolarCodec,
    RCSCANPolarCodec,
)
from ..polar_codes.fast_scan import FastSCANCodec
from ..polar_codes.g_fast_scan import GFastSCANCodec
from . import functions, http


class CodeTypes:
    """Code types"""
    FAST_SSC = 'fast-ssc'
    RC_SCAN = 'rc_scan'
    G_FAST_SSC = 'g-fast-ssc'
    FAST_SCAN = 'fast-scan'
    G_FAST_SCAN = 'g-fast-scan'
    ALL = [
        FAST_SSC,
        RC_SCAN,
        G_FAST_SSC,
        FAST_SCAN,
        G_FAST_SCAN,
    ]
    SCAN = [
        RC_SCAN,
        FAST_SCAN,
        G_FAST_SCAN,
    ]
    GENERALIZED = [
        G_FAST_SSC,
        G_FAST_SCAN,
    ]


class ChannelTypes:
    SIMPLE_BPSK = 'simple-bpsk'


CODE_MAP = {
    CodeTypes.FAST_SSC: FastSSCPolarCodec,
    CodeTypes.RC_SCAN: RCSCANPolarCodec,
    CodeTypes.G_FAST_SSC: GFastSSCPolarCodec,
    CodeTypes.FAST_SCAN: FastSCANCodec,
    CodeTypes.G_FAST_SCAN: GFastSCANCodec,
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


def simulate_from_params(url: str):
    """Simulate polar code chain using remote params."""
    status_code, experiment = http.get_params(url=url)
    if status_code != 200:
        print('No experiment data!')
        return

    channel_type = experiment.pop('channel_type')
    code_id = experiment.pop('code_id')
    code_type = experiment.pop('code_type')
    snr = experiment.pop('snr')
    messages = experiment.pop('messages')
    # Pop `type` to prevent problems with initialization
    cls = experiment.pop('type')

    result = simulate(
        code_type=code_type,
        channel_type=channel_type,
        snr=snr,
        messages=messages,
        code_params=experiment,
    )

    result_log = (
        f'Result: {result}\n'
        f'{code_type.upper()} ({experiment["N"]},{experiment["K"]})'
    )
    if code_type in CodeTypes.SCAN:
        result_log += f', I = {experiment["I"]}'
    if code_type in CodeTypes.GENERALIZED:
        result_log += f', AF = {experiment["AF"]}'
    print(result_log)

    resp = http.save_result(
        url=url,
        result=result,
        code_id=code_id,
        code_type=code_type,
        channel_type=channel_type,
        cls=cls,
    )
    print(f'Status {resp.status_code}: {resp.json()}')


def simulate_multi_core(experiments: int, url: str):
    """Simulate polar code chain using multiple cores."""
    workers = multiprocessing.cpu_count()
    print(f'Workers: {workers}; Number of experiments: {experiments}')

    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        run_tasks = {
            ex.submit(simulate_from_params, *(url, )): (url, )
            for _ in range(experiments)
        }
        for future in futures.as_completed(run_tasks):
            try:
                future.result()
            except Exception as exc:
                print(exc)
