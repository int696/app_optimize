from datetime import datetime
import requests
import pandas as pd
from utils.create_file_and_path import Util


class WeatherForecast:

    def __init__(self):
        self.__params = None
        self.__headers = None
        self.set_param()
        self.name_file = Util()
        self.status_code = None

    def set_param(self, station='1', var_station='mpei', var_nwp_provider='icon'):
        self.__headers = {
            'accept': 'application/json',
        }

        self.__params = {
            'orgId': station,
            'var-station': var_station,
            'var-nwp_provider': var_nwp_provider,
        }

    def get_json(self):
        address = f"https://api.pvforecast.ru/stations/" \
                  f"{self.__params['var-station']}/" \
                  f"{self.__params['var-nwp_provider']}"

        response = requests.get(address, headers=self.__headers)
        if response.status_code == 200:
            try:
                print(200)
                self.status_code = 200
                json_data = response.json()
                data = self.__get_data()
                self.name_file.create_json("weather.json", json_data[data:24+data])
            except ValueError:
                print("Сервер вернул некорректный JSON")
        else:
            print(f"Запрос завершился с кодом ошибки {response.status_code}")

    @staticmethod
    def __get_data():
        return int(datetime.now().strftime("%H"))


    @staticmethod
    def get_param_weather(hour=1, *param):
        data_json = Util().open_json("weather.json")
        if param:
            df = pd.json_normalize(data_json).iloc[hour].loc[list(param)]
        else:
            df = pd.json_normalize(data_json).iloc[hour]
        return df
