import math
from datetime import datetime

import pulp
import pandas as pd
from pulp import PULP_CBC_CMD
from utils.create_file_and_path import Util


class Optimize:
    """
    Класс предназначен для проведения оптимизации состава ДЭС по целевой мощности, с помощью солвера СВС, работает по
    принципу ЛП с поиском минимума и учетом ограничений.
    """

    def __init__(self):

        # self.flag_test_load = False

        self.consumption_all = None
        self.power_all = None
        self.name_file = None
        self.flag_save = False
        self.cons_idx = []
        self.output_L = None
        self.num_engines = 6
        self.engine_min_max = None
        self.num_points = None
        self.output_W = None
        self.list_dgu = []

    def optimize(self, excluded_engines, target_w=0):

        assignments = pulp.LpVariable.matrix(
            name='asn', cat=pulp.LpBinary,
            indices=(range(self.num_engines), range(self.num_points)),
        )
        prob = pulp.LpProblem(name='diesel_generation', sense=pulp.LpMinimize)
        fuel_cost = pulp.LpAffineExpression()
        for engine, engine_group in enumerate(assignments):
            if excluded_engines[engine] != 0:
                fuel_cost += pulp.lpDot(engine_group, self.output_L.iloc[:, engine])
        # prob.objective += fuel_cost
        total_output = pulp.LpAffineExpression()
        for engine, engine_group in enumerate(assignments):
            if excluded_engines[engine] != 0:
                prob.addConstraint(name=f'engine_excl_{engine}', constraint=pulp.lpSum(engine_group) <= 1)
                total_output += pulp.lpDot(engine_group, self.output_W.iloc[:, engine])
                prob.objective += pulp.lpDot(engine_group, self.output_L.iloc[:, engine])
        prob += total_output == target_w
        prob.solve(PULP_CBC_CMD(msg=False))
        # prob.writeLP("TaxiAssignmentProblem.lp")
        try:
            assert prob.status == pulp.LpStatusOptimal
        except Exception as e:
            print(e)
        self.cons_idx = [
            next((i for i, var in enumerate(engine_group) if var.value() is not None and var.value() > 0.5), None)
            for engine_group in assignments
        ]
        self.list_dgu = []

        b = 0
        p = 0
        d = ''
        for idx, cons_idx in enumerate(self.cons_idx):
            if cons_idx is not None:
                print(f"Дизель {idx + 1} включен, его мощность: {self.output_W.iloc[cons_idx, idx]}", end=' ')
                print(f"его расход: {self.output_L.iloc[cons_idx, idx]}")
                self.list_dgu.append([idx, self.output_W.iloc[cons_idx, idx], self.output_L.iloc[cons_idx, idx]])
                b += self.output_L.iloc[cons_idx, idx]
                p += self.output_W.iloc[cons_idx, idx]
                d += f'{idx + 1} '
            else:
                self.list_dgu.append([idx, 0, 0])
        if self.flag_save:
            column = [b, p, d, datetime.now()]
            Util().open_csv(self.name_file, mode='a', data=column)
        print("Суммарный расход:", b)
        print('================================================================================')

    def init_optimize(self, operator, k):
        power_dgu = operator.get_power_dgu()

        self.power_all = pd.DataFrame(power_dgu).set_index('No. of item')
        self.output_W = self.generate_matrix(self.power_all)
        self.num_points = int((self.power_all.max(axis=1).iloc[-1] - self.power_all.min(axis=1).iloc[0]) / k)
        consumption_dgu = operator.get_consumption_dgu()
        self.consumption_all = pd.DataFrame(consumption_dgu).set_index('No. of item')
        self.output_L = self.generate_matrix(self.consumption_all)


    @staticmethod
    def generate_matrix(data):
        output = {}
        for c, r in data.items():
            output[c] = []
            for c_1, r_1 in r.items():

                try:
                    if r_1 is None or math.isnan(r_1):
                        output[c].append(data[c].min(axis=0))
                        pass
                    else:
                        output[c].append(r_1)
                except Exception as e:
                    print(e)

        output = pd.DataFrame(output)
        return output

    def save_optimize(self, name_file, column):
        self.name_file = f'{name_file}.csv'
        self.flag_save = True
        Util().open_csv(self.name_file, mode='w', data=column)

