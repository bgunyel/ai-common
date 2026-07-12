"""
* This file is for active development of the package and hence,  it is a living file.

* At any time, you can consult this file for the usage of package components.
* However, the contents of this file may change  abruptly, at any time.
* For permanent usage examples, please see the <tests> folder
"""

import datetime
import os
import rich
import time
from uuid import uuid4

from config import get_settings, get_llm_config


def main():

    settings = get_settings()
    llm_config = get_llm_config()

    os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY.get_secret_value()
    os.environ['LANGSMITH_TRACING'] = settings.LANGSMITH_TRACING
    os.environ['LANGSMITH_PROJECT'] = settings.APPLICATION_NAME.lower()


    dummy = -32


if __name__ == '__main__':
    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))

    config_settings = get_settings()
    print(f'{config_settings.APPLICATION_NAME} started at {time_now}')
    time1 = time.time()
    main()
    time2 = time.time()

    time_now = datetime.datetime.now().replace(microsecond=0).astimezone(
        tz=datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3'))
    print(f'{config_settings.APPLICATION_NAME} finished at {time_now}')
    print(f'{config_settings.APPLICATION_NAME} took {(time2 - time1):.2f} seconds')
