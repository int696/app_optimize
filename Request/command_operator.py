from Connected.connection_db import add_user


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
        excluded_engines = [i['available_dgu'] for i in excluded_engines]
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

    def get_param_des(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Параметры_ДЭС")
        n_dgu = cursor.fetchall()[0]['N_dgu']
        return n_dgu

    def get_power_dgu_max_min(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Максимальные_минимальные_мощности_ДГУ")
        power_dgu = list(map(lambda x: x['DGU_pmax'], cursor.fetchall()))
        return power_dgu

    def get_Kb_Bb(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Коэффициенты_K_B")
        K_list = cursor.fetchall()
        Kb = list(map(lambda x: x['Kb'], K_list))
        Bb = list(map(lambda x: x['Bb'], K_list))
        return Kb, Bb

    def get_startup_cost(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Затраты_на_холодный_пуск_ДГУ")
        startup_cost = list(map(lambda x: x['startup_cost'], cursor.fetchall()))
        return startup_cost

    def get_shutdown_cost(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Затраты_на_остановку_ДГУ")
        shutdown_cost = list(map(lambda x: x['shutdown_cost'], cursor.fetchall()))
        return shutdown_cost

    def get_fuel_price(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Цена_диз_топлива")
        fuel_price = list(map(lambda x: x['fuel_price'], cursor.fetchall()))
        return fuel_price[0]

    def get_param_inv(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Параметры_инверторов")
        K_list = cursor.fetchall()
        ESS_inv = list(map(lambda x: x['ESS_inv'], K_list))
        PV = list(map(lambda x: x['PV'], K_list))
        return *ESS_inv, *PV

    def get_param_SNE(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Параметры_СНЭ")
        K_list = cursor.fetchall()
        battery_capacity = list(map(lambda x: x['battery_capacity'], K_list))
        SOCmin = list(map(lambda x: x['SOCmin'], K_list))
        SOCmax = list(map(lambda x: x['SOCmax'], K_list))
        Eff_Bat = list(map(lambda x: x['Eff_Bat'], K_list))
        soc_after = list(map(lambda x: x['soc_after'], K_list))
        return *battery_capacity, *SOCmin, *SOCmax, *Eff_Bat, *soc_after

    def get_ess_availability_state(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Доступность_инверторов_СНЭ_на_момент_до_начала_расчета")
        K_list = cursor.fetchall()
        ess_availability_state = list(map(lambda x: x['ess_availability_state'], K_list))
        return ess_availability_state

    def get_energy_storage_system(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM energy_storage_system")
        data = cursor.fetchall()[0]
        return data

    def get_up_before(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Число_часов_работы_ДГУ_на_момент_до_начала_расчета")
        K_list = cursor.fetchall()
        up_before = list(map(lambda x: x['up_before'], K_list))
        return up_before

    def get_down_before(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Число_часов_простоя_ДГУ_на_момент_до_начала_расчета")
        K_list = cursor.fetchall()
        down_before = list(map(lambda x: x['down_before'], K_list))
        return down_before

    def get_control_dgu_new(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM control_dgu_new")
        K_list = cursor.fetchall()
        control_dgu_new = list(map(lambda x: x['control_dgu'], K_list))
        return control_dgu_new

    def get_min_up_time(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Минимальное_допустимое_число_последовательных_часов_работы_ДГУ")
        K_list = cursor.fetchall()
        min_up_time = list(map(lambda x: x['min_up_time'], K_list))
        return min_up_time

    def get_min_down_time(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Минимальное_допустимое_число_последовательных_часов_простоя_ДГУ")
        K_list = cursor.fetchall()
        min_down_time = list(map(lambda x: x['min_down_time'], K_list))
        return min_down_time

    def delete_optimize_24(self):
        cursor = self.connect.cursor()
        cursor.execute(f"UPDATE Оптимизация_24_часа "
                       f"SET "
                       f"D1 = NULL, "
                       f"D2 = NULL, "
                       f"D3 = NULL, "
                       f"D4 = NULL, "
                       f"D5 = NULL, "
                       f"D6 = NULL, "
                       f"CH1 = NULL, "
                       f"dCH1 = NULL, "
                       f"CH2 = NULL, "
                       f"dCH2 = NULL, "
                       f"SOC1 = NULL, "
                       f"SOC2 = NULL, "
                       f"PV1 = NULL, "
                       f"PV2 = NULL, "
                       f"PV3 = NULL, "
                       f"PV4 = NULL, "
                       f"PV5 = NULL, "
                       f"PV6 = NULL, "
                       f"PV7 = NULL, "
                       f"Diesel = NULL, "
                       f"ESS = NULL, "
                       f"PV = NULL, "
                       f"Load = NULL")
        cursor.close()

    import pandas as pd

    def update_optimize_24(self, data):
        cursor = self.connect.cursor()
        query = ("""
            UPDATE Оптимизация_24_часа
            SET 
            D1 = %s,
            D2 = %s,
            D3 = %s,
            D4 = %s,
            D5 = %s,
            D6 = %s,
            CH1 = %s,
            dCH1 = %s,
            CH2 = %s,
            dCH2 = %s,
            SOC1 = %s,
            SOC2 = %s,
            PV1 = %s,
            PV2 = %s,
            PV3 = %s,
            PV4 = %s,
            PV5 = %s,
            PV6 = %s,
            PV7 = %s,
            Diesel = %s,
            ESS = %s,
            PV = %s,
            Load_plan = %s
            WHERE id = %s
        """)

        data_list = data.values.tolist()
        data_with_index = [row + [data.index[i] + 1] for i, row in enumerate(data_list)]
        print(data_with_index)
        cursor.executemany(query, data_with_index)
        self.connect.commit()

    def get_power_optimize_24(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM Оптимизация_24_часа WHERE id ='1'")
        data = cursor.fetchall()[0]
        return data

    def get_power_optimize_real(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT * FROM ДГУ_текущее_состояние_параметров")
        data = cursor.fetchall()
        data = sum(list(map(lambda x: x['P_sum_DG'], data)))
        return data


if __name__ == '__main__':
    connect = add_user()
    operator = Command(connect)

    print(operator.get_power_optimize_real())


