import time
import datetime
from Connected.connection_db import add_user
from Connected.contact_mqtt import connection
from Optimize_real.optimize import Optimize
from Request.command_operator import Command


def timer(step, operator):
    list_f = []
    print(f'Усреднение частоты старт, {datetime.datetime.now().second}:{datetime.datetime.now().microsecond}')
    start_time = time.time()
    while time.time() - start_time < step:
        list_f.append(operator.get_energy_storage_system()['F'])

    print(f'Усреднение частоты стоп, {datetime.datetime.now().second}:{datetime.datetime.now().microsecond}')
    return sum(list_f) / len(list_f)


def forecast(optimizator, operator):
    settings = operator.get_setting()
    optimizator.init_optimize(operator, settings["Step_power"])
    excluded_engines = operator.get_excluded_engines()



    while True:
        if operator.check_connections("start_stop_optimizator"):

            f = timer(settings['time_step'], operator)
            power_forecast = operator.get_power_optimize_24()['Diesel']
            power_real = operator.get_power_optimize_real()
            operator.delete_forecast_power()
            # optimizator.optimize(excluded_engines, [power_forecast + 35, power_real], settings, f)
            optimizator.optimize(excluded_engines, [35, 35], settings, f)
            # operator.update_forecast_power(h, lo+35, optimizator.list_dgu)
            operator.update_excluded_engines(optimizator.list_dgu)


def init_start():
    optimizator = Optimize()
    connect = add_user()
    operator = Command(connect)
    forecast(optimizator, operator)


if __name__ == '__main__':
    init_start()
