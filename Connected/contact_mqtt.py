import paho.mqtt.client as mqtt
import configparser

"""
Дефолтные методы для подключения к Connected
"""

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Подключение к MQTT СЕРВЕРУ прошло успешно")
        print("------------------------------------------------------------------")
        client.subscribe('#')
    else:
        print("Ошибка при подключении к MQTT СЕРВЕРУ")


def connection(config_path="utils/setting.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)

    user = config["MQTT"]["USER"]
    password = config["MQTT"]["PASSWORD"]
    host = config["MQTT"]["MQTT_HOST"]
    port = int(config["MQTT"]["MQTT_PORT"])
    interval = int(config["MQTT"]["MQTT_KEEPALIVE_INTERVAL"])

    mqttc = mqtt.Client()
    mqttc.username_pw_set(username=user, password=password)
    mqttc.on_connect = on_connect
    mqttc.connect(host, port, interval)
    mqttc.loop_start()

    return mqttc

