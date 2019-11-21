import multiprocessing
from concurrent import futures
from datetime import datetime
from functools import partial

from modelling.channels import VerificationChannel
from modelling.functions import generate_simulation_parameters, simulation_task
from modelling.mongo import DB_NAME


def run_model(workers, task, list_of_parameters):
    """"""
    print('Start execution at', datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        run_tasks = {ex.submit(task, *params): params for params in
                     list_of_parameters}
        for future in futures.as_completed(run_tasks):
            try:
                future.result()
            except Exception as exc:
                print(exc)

    print('Finish execution at', datetime.now().strftime('%H:%M:%S %d-%m-%Y'))


def executor(code_class, codeword_length, code_rates, snr_range,
             task_repetitions, messages_per_task, collection_name,
             db_name=DB_NAME, channel_class=VerificationChannel, workers=None):
    """"""
    list_of_task_parameters = generate_simulation_parameters(
        code_cls=code_class,
        channel_cls=channel_class,
        N=codeword_length,
        code_rates=code_rates,
        snr_range=snr_range,
        repetitions=task_repetitions
    )
    task = partial(
        simulation_task,
        db_name=db_name,
        collection=collection_name,
        messages=messages_per_task
    )
    workers = workers or multiprocessing.cpu_count() - 2

    run_model(workers, task, list_of_task_parameters)
