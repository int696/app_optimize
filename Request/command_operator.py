

class Command:

    def __init__(self, connect):
        self.connect = connect

    def check_connections(self, column):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM control_signal WHERE id ='1'")
        start_stop = cursor.fetchone()[column]
        cursor.close()
        return start_stop

    def get_power_dgu(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM power_dgu")
        power_dgu = cursor.fetchall()
        cursor.close()
        return power_dgu

    def get_consumption_dgu(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM consumption_dgu")
        consumption_dgu = cursor.fetchall()
        cursor.close()
        return consumption_dgu

    def get_setting(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM settings")
        setting = cursor.fetchall()[0]
        cursor.close()
        return setting

    def get_excluded_engines(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM available_dgu_new")
        excluded_engines = cursor.fetchall()
        cursor.close()
        return excluded_engines

    def update_excluded_engines(self, available_dgu):
        cursor = self.connect.cursor()
        for dgu in available_dgu:
            if dgu[1] != 0:
                cursor.execute(f"UPDATE control_dgu_new SET control_dgu = 1 WHERE id = {dgu[0] + 1}")
            else:
                cursor.execute(f"UPDATE control_dgu_new SET control_dgu = 0 WHERE id = {dgu[0] + 1}")
        cursor.close()

    def update_forecast_power(self, h, power, list_dgu):
        cursor = self.connect.cursor()
        for n, p, _ in list_dgu:

            cursor.execute(f"UPDATE forecast_power "
                           f"SET "
                           f"power_kW = {power}, "
                           f"dgu_{n + 1} = {p} "
                           f"WHERE id = {h}")
        cursor.close()

    def delete_forecast_power(self):
        cursor = self.connect.cursor()
        cursor.execute(f"UPDATE forecast_power "
                       f"SET "
                       f"power_kW = NULL, "
                       f"dgu_1 = NULL, "
                       f"dgu_2 = NULL, "
                       f"dgu_3 = NULL, "
                       f"dgu_4 = NULL, "
                       f"dgu_5 = NULL, "
                       f"dgu_6 = NULL")
        cursor.close()

    def update_current_power(self, power):
        cursor = self.connect.cursor()
        cursor.execute(f"UPDATE current_power SET current_power = {power} WHERE id = 1")
        cursor.close()

    def get_load(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM load_forecast")
        load = list(map(lambda lo: lo['power_kW'], cursor.fetchall()))[:-1]
        return load
