import configparser

import pymysql
from pymysql.cursors import DictCursor

from utils.create_file_and_path import Util


def add_user():
    # Получаем данные пользователя из функции
    config = configparser.ConfigParser()
    # config.read("utils/setting.ini")
    # host = config['BD']['HOST']
    # user = config['BD']['USER']
    # password = config['BD']['PASSWORD']
    # BD = config['BD']['BD_NAME']
    try:
        connect = pymysql.connect(
            host="10.2.173.169",
            port=3306,
            user="user1",
            password="123",
            cursorclass=DictCursor,
            database="test",
            autocommit=True
        )

        return connect
    except Exception as e:
        print("Подключение не удалось")
        print(e)
        raise
