import configparser
import csv
import json
import os
from sys import platform
import logging


class Util:

    @staticmethod
    def get_data_path(name_file, path="\\utils\\"):
        if platform == 'win32' or platform == 'win64':
            return f"{path}{name_file}"
        elif platform == 'linux' or platform == 'linux2':
            return f"/utils/{name_file}"

    def open_json(self, name_file):
        try:
            current_script_path = os.path.abspath(__file__)
            path = self.get_data_path(name_file)
            project_root_path = os.path.dirname(os.path.dirname(current_script_path)) + path
            with open(project_root_path, 'r') as json_file:
                data = json.load(json_file)
            return data
        except Exception as e:
            print(f"Ошибка открытия {name_file}: {e}")

    def create_json(self, name_file, data):
        try:
            current_script_path = os.path.abspath(__file__)
            path = self.get_data_path(name_file)
            project_root_path = os.path.dirname(os.path.dirname(current_script_path)) + path
            with open(project_root_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        except Exception as e:
            print(f"Ошибка создания {name_file}: {e}")

    def open_csv(self, name_file, mode, data):
        current_script_path = os.path.abspath(__file__)
        path = self.get_data_path(name_file)
        project_root_path = os.path.dirname(os.path.dirname(current_script_path)) + path
        with open(project_root_path, mode=mode, encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=";")
            data_to_add = data
            writer.writerow(data_to_add)

    def create_log(self, mode):
        current_script_path = os.path.abspath(__file__)
        path = self.get_data_path("py_log.log")
        project_root_path = os.path.dirname(os.path.dirname(current_script_path)) + path
        logging.basicConfig(level=logging.INFO, filename=project_root_path, filemode=mode,
                            format="%(asctime)s %(levelname)s %(message)s")

    def config_pars(self, name):
        config = configparser.ConfigParser()
        project_root_path = self.get_data_path(name)
        config.read(project_root_path)
        return config
