import time

from Connected.connection_db import add_user
from Connected.contact_mqtt import connection
from Optimize_real.optimize import Optimize
from Request.command_operator import Command


def forecast(optimizator, operator):
    excluded_engines = operator.get_excluded_engines()
    excluded_engines = [i['available_dgu'] for i in excluded_engines]
    # print(excluded_engines)
    while True:
        if operator.check_connections("start_stop_optimizator"):

            load = operator.get_load()
            operator.delete_forecast_power()
            for h, lo in enumerate(load, start=1):
                optimizator.optimize(excluded_engines, lo+35)
                operator.update_forecast_power(h, lo+35, optimizator.list_dgu)
                operator.update_excluded_engines(optimizator.list_dgu)
                time.sleep(2)
                # print(lo+35, h)


def init_start():
    optimizator = Optimize()
    connect = add_user()
    operator = Command(connect)
    settings = operator.get_setting()
    optimizator.init_optimize(operator, settings["Step_power"])
    forecast(optimizator, operator)


if __name__ == '__main__':
    init_start()
