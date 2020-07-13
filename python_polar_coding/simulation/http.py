import requests


def prepare(
        url,
        snr_range,
        required_messages,
        codes,
        per_experiment=1000,
):
    params = {
        'codes': codes,
        'snr_range': snr_range,
        'required_messages': required_messages,
        'per_experiment': per_experiment,
    }
    try:
        return requests.post(f'{url}/prepare', json=params)
    except Exception as e:
        print(e)


def get_params(url):
    try:
        resp = requests.put(f'{url}/get-params', json={})
        return resp.status_code, resp.json()
    except Exception as e:
        print(e)


def save_result(url, result, code_id, code_type, cls, channel_type):
    result.update({'route_params': {
        'code_id': code_id,
        'code_type': code_type,
        'channel_type': channel_type,
        'type': cls,
    }})
    try:
        return requests.post(f'{url}/save-result', json=result)
    except Exception as e:
        print(e)
