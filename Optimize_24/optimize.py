import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
import pvlib
import math
#from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import time
#import pypsa
from pyomo.environ import *
import requests
from io import StringIO
import math
import numpy as np

T1 = 0
T2 = T1 + 24
HOY = np.array([t for t in range(T1, T2)])
HOD = np.array([t for t in range(0, 24)])
T = np.array([t for t in range(T1, T2)])


Load8760 = pd.read_csv('LLO_LoadForecast.csv', sep=',', header=0)
# ..."Вырезание" из данных за один год ближайших 24-х часов...
Load = Load8760['Load'].values

NPV_forecast = pd.read_csv('LLO_PVPowerForecast.csv', sep=',', header=0)
# ...Получение прогноза мощности СЭС...
PV = NPV_forecast['P'].values

fig, ax = plt.subplots(2, 1, figsize=(18, 16))

ax[0].bar(HOY, Load[HOD], color='blue')
ax[0].set_xlabel('Номер часа')
ax[0].set_ylabel('Нагрузка, кВт')
ax[0].grid(linestyle='--')

ax[1].bar(HOY, PV[HOD], color='yellow')
ax[1].set_xlabel('Номер часа')
ax[1].set_ylabel('Выработка СЭС, кВт')
ax[1].grid(linestyle='--')

fig.tight_layout(pad=2)

plt.show()

#количество ДГУ
Ndgu = 1
N = np.array([n for n in range(0, Ndgu)])

#ммаксимальные мощности ДГУ
DGU1_pmax = 12 #кВт
DGU1_pmin = 0.3 * DGU1_pmax #минимально-допустимая мощность ДГУ

DGU2_pmax = 0 #кВт
DGU2_pmin = 0.1 * DGU2_pmax

DGU3_pmax = 0 #кВт
DGU3_pmin = 0.1 * DGU3_pmax

DGU4_pmax = 0 #кВт
DGU4_pmin = 0.1 * DGU4_pmax

DGU5_pmax = 0 #кВт
DGU5_pmin = 0.1 * DGU5_pmax

DGU6_pmax = 0 #кВт
DGU6_pmin = 0.1 * DGU6_pmax

d1_availability_state = 1
d2_availability_state = 0
d3_availability_state = 0
d4_availability_state = 0
d5_availability_state = 0
d6_availability_state = 0

# Кривые расхода топлива ДГУ

# Диапазон построения кривых расхода топлива ДГУ
p1 = np.arange(0, DGU1_pmax + 1)
p2 = np.arange(0, DGU2_pmax + 1)
p3 = np.arange(0, DGU3_pmax + 1)
p4 = np.arange(0, DGU4_pmax + 1)
p5 = np.arange(0, DGU5_pmax + 1)
p6 = np.arange(0, DGU6_pmax + 1)

# Коэффициенты K и B прямой линии, аппроксимирующей (абсолютную) расходную характеристику ДГУ (из паспортов ДГУ)
Kb1 = 0.609
Kb2 = 0.25
Kb3 = 0.24
Kb4 = 0.23
Kb5 = 0.22
Kb6 = 0.21

Bb1 = -0.48
Bb2 = 0
Bb3 = 0
Bb4 = 0
Bb5 = 0
Bb6 = 0

# Кривые абсолютного расхода топлива ДГУ
DGU1_FuelCurve = Kb1 * p1 + Bb1
DGU2_FuelCurve = Kb2 * p2 + Bb2
DGU3_FuelCurve = Kb3 * p3 + Bb3
DGU4_FuelCurve = Kb4 * p4 + Bb4
DGU5_FuelCurve = Kb5 * p5 + Bb5
DGU6_FuelCurve = Kb6 * p6 + Bb6

DGU1_fuel = DGU1_FuelCurve
DGU2_fuel = DGU2_FuelCurve
DGU3_fuel = DGU3_FuelCurve
DGU4_fuel = DGU4_FuelCurve
DGU5_fuel = DGU5_FuelCurve
DGU6_fuel = DGU6_FuelCurve

# Построение расходных характеристик только для тех ДГУ, что доступны к работе
fig, ax = plt.subplots(1, 1)
if d1_availability_state == 1: ax.plot(DGU1_FuelCurve, label='ДГУ #1')
if d2_availability_state == 1: ax.plot(DGU2_FuelCurve, label='ДГУ #2')
if d3_availability_state == 1: ax.plot(DGU3_FuelCurve, label='ДГУ #3')
if d4_availability_state == 1: ax.plot(DGU4_FuelCurve, label='ДГУ #4')
if d5_availability_state == 1: ax.plot(DGU5_FuelCurve, label='ДГУ #5')
if d6_availability_state == 1: ax.plot(DGU6_FuelCurve, label='ДГУ #6')

xmax = max(DGU1_pmax, DGU2_pmax, DGU3_pmax, DGU4_pmax, DGU5_pmax, DGU6_pmax)
ax.set_xlim(0, xmax * 1.05)
ymax = max(DGU1_FuelCurve.max(), DGU2_FuelCurve.max(), DGU3_FuelCurve.max(), DGU4_FuelCurve.max(), DGU5_FuelCurve.max(),
           DGU6_FuelCurve.max())
ax.set_ylim(0, ymax * 1.05)
ax.set_xlabel('Мощность, кВт')
ax.set_ylabel('Расход топлива, л/ч')
ax.legend(loc='best')
ax.grid()

# затраты на холодный пуск ДГУ
d1_startup_cost = 0
d2_startup_cost = 0
d3_startup_cost = 0
d4_startup_cost = 0
d5_startup_cost = 0
d6_startup_cost = 0

# затраты на остановку ДГУ
d1_shutdown_cost = 0
d2_shutdown_cost = 0
d3_shutdown_cost = 0
d4_shutdown_cost = 0
d5_shutdown_cost = 0
d6_shutdown_cost = 0

# цена диз.топлива
Fuel_price = 60  # руб/л

PV_inv1 = 7
PV_inv2 = 7
PV_inv3 = 7
PV_inv4 = 7
PV_inv5 = 7
PV_inv6 = 7
PV_inv7 = 7

# Мощность батарейных инверторов
ESS_inv = 30

BatteryCapacity = 19.2 # Ёмкость СНЭ в кВтч
SOCmin = 50           # Минимально допустимый уровень заряда СНЭ, %
SOCmax = 100          # Максимально допустимый уровень заряда СНЭ, %
Eff_Bat = 1.00        # КПД СНЭ (КПД аккумуляторов и инверторов, коэффициент саморазряда пока не включен)

# Доступность инверторов СНЭ на момент до начала расчета (0 - инвертор СНЭ недоступен)
ess1_availability_state = 1
ess2_availability_state = 1

# Уровень заряда СНЭ на момент начала расчета
# ПРИМЕЧАНИЕ: значение должно быть получено из БД в момент начала работы Палнировщика
soc1_before = 51
soc2_before = 51

# Настраиваемый уровень заряда СНЭ на конец расчетного периода (если нужно зарядить/разрядить АКБ принудительно)
soc1_after = 100
soc2_after = 100

m = pyo.ConcreteModel()

output = []

#Граничные условия ДГУ

#число часов работы ДГУ на момент до начала расчета
d1_up_before = 0
d2_up_before = 0
d3_up_before = 0
d4_up_before = 0
d5_up_before = 0
d6_up_before = 0

#число часов простоя ДГУ на момент до начала расчета
d1_down_before = 0
d2_down_before = 0
d3_down_before = 0
d4_down_before = 0
d5_down_before = 0
d6_down_before = 0

# Статус работы ДГУ на момент начала расчета
# ПРИМЕЧАНИЕ: значение должно быть получено из БД в момент начала работы Палнировщика
u1_start = 0
u2_start = 0
u3_start = 0
u4_start = 0
u5_start = 0
u6_start = 0

# Минимальное допустимое число последовательных часов простоя ДГУ
d1_min_up_time = 0
d2_min_up_time = 0
d3_min_up_time = 0
d4_min_up_time = 0
d5_min_up_time = 0
d6_min_up_time = 0

# Минимальное допустимое число последовательных часов простоя ДГУ
d1_min_down_time = 0
d2_min_down_time = 0
d3_min_down_time = 0
d4_min_down_time = 0
d5_min_down_time = 0
d6_min_down_time = 0

def balance(model, i):
    return m.x1[i] + m.x2[i] + m.x3[i] + m.x4[i] + m.x5[i]+ m.x6[i] + m.bat1_dch[i] - m.bat1_ch[i] + m.bat2_dch[i] - m.bat2_ch[i] + m.PV1[i] + m.PV2[i] + m.PV3[i] + m.PV4[i] + m.PV5[i] + m.PV6[i] + m.PV7[i] == (np.asarray(Load)[i])


def d1_min_up_time_c(model, i):
    if i == T1:
        return (d1_up_before - d1_min_up_time) * (u1_start - m.u1[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u1[i - 1])
        for j in range(T1, i):
            uj.append(m.u1[j])
        return ((sum(uj) - d1_min_up_time) * (m.u1[i - 1] - m.u1[i])) >= 0
    else:
        return ((m.u1[i - 1] + m.u1[i - 2] + m.u1[i - 3] - d1_min_up_time) * (m.u1[i - 1] - m.u1[i])) >= 0


def d2_min_up_time_c(model, i):
    if i == T1:
        return (d2_up_before - d2_min_up_time) * (u2_start - m.u2[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u2[i - 1])
        for j in range(T1, i):
            uj.append(m.u2[j])
        return ((sum(uj) - d2_min_up_time) * (m.u2[i - 1] - m.u2[i])) >= 0
    else:
        return ((m.u2[i - 1] + m.u2[i - 2] + m.u2[i - 3] - d2_min_up_time) * (m.u2[i - 1] - m.u2[i])) >= 0


def d3_min_up_time_c(model, i):
    if i == T1:
        return (d3_up_before - d3_min_up_time) * (u3_start - m.u3[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u3[i - 1])
        for j in range(T1, i):
            uj.append(m.u3[j])
        return ((sum(uj) - d3_min_up_time) * (m.u3[i - 1] - m.u3[i])) >= 0
    else:
        return ((m.u3[i - 1] + m.u3[i - 2] + m.u3[i - 3] - d3_min_up_time) * (m.u3[i - 1] - m.u3[i])) >= 0


def d4_min_up_time_c(model, i):
    if i == T1:
        return (d4_up_before - d4_min_up_time) * (u4_start - m.u4[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u4[i - 1])
        for j in range(T1, i):
            uj.append(m.u4[j])
        return ((sum(uj) - d4_min_up_time) * (m.u4[i - 1] - m.u4[i])) >= 0
    else:
        return ((m.u4[i - 1] + m.u4[i - 2] + m.u4[i - 3] - d4_min_up_time) * (m.u4[i - 1] - m.u4[i])) >= 0


def d5_min_up_time_c(model, i):
    if i == T1:
        return (d5_up_before - d5_min_up_time) * (u5_start - m.u5[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u5[i - 1])
        for j in range(T1, i):
            uj.append(m.u5[j])
        return ((sum(uj) - d5_min_up_time) * (m.u5[i - 1] - m.u5[i])) >= 0
    else:
        return ((m.u5[i - 1] + m.u5[i - 2] + m.u5[i - 3] - d5_min_up_time) * (m.u5[i - 1] - m.u5[i])) >= 0


def d6_min_up_time_c(model, i):
    if i == T1:
        return (d6_up_before - d6_min_up_time) * (u6_start - m.u6[i]) >= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u6[i - 1])
        for j in range(T1, i):
            uj.append(m.u6[j])
        return ((sum(uj) - d6_min_up_time) * (m.u6[i - 1] - m.u6[i])) >= 0
    else:
        return ((m.u6[i - 1] + m.u6[i - 2] + m.u6[i - 3] - d6_min_up_time) * (m.u6[i - 1] - m.u6[i])) >= 0


def d1_min_down_time_c(model, i):
    if i == T1:
        return (d1_down_before + d1_min_down_time) * (-u1_start + m.u1[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u1[i - 1])
        for j in range(T1, i):
            uj.append(m.u1[j])
        return ((sum(uj) - 2 + d1_min_down_time) * (-m.u1[i - 1] + m.u1[i])) <= 0
    else:
        return ((m.u1[i - 1] + m.u1[i - 2] + m.u1[i - 3] - 3 + d1_min_down_time) * (-m.u1[i - 1] + m.u1[i])) <= 0


def d2_min_down_time_c(model, i):
    if i == T1:
        return (d2_down_before + d2_min_down_time) * (-u2_start + m.u2[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u2[i - 1])
        for j in range(T1, i):
            uj.append(m.u2[j])
        return ((sum(uj) - 2 + d2_min_down_time) * (-m.u2[i - 1] + m.u2[i])) <= 0
    else:
        return ((m.u2[i - 1] + m.u2[i - 2] + m.u2[i - 3] - 3 + d2_min_down_time) * (-m.u2[i - 1] + m.u2[i])) <= 0


def d3_min_down_time_c(model, i):
    if i == T1:
        return (d3_down_before + d3_min_down_time) * (-u3_start + m.u3[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u3[i - 1])
        for j in range(T1, i):
            uj.append(m.u3[j])
        return ((sum(uj) - 2 + d3_min_down_time) * (-m.u3[i - 1] + m.u3[i])) <= 0
    else:
        return ((m.u3[i - 1] + m.u3[i - 2] + m.u3[i - 3] - 3 + d3_min_down_time) * (-m.u3[i - 1] + m.u3[i])) <= 0


def d4_min_down_time_c(model, i):
    if i == T1:
        return (d4_down_before + d4_min_down_time) * (-u4_start + m.u4[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u4[i - 1])
        for j in range(T1, i):
            uj.append(m.u4[j])
        return ((sum(uj) - 2 + d4_min_down_time) * (-m.u4[i - 1] + m.u4[i])) <= 0
    else:
        return ((m.u4[i - 1] + m.u4[i - 2] + m.u4[i - 3] - 3 + d4_min_down_time) * (-m.u4[i - 1] + m.u4[i])) <= 0


def d5_min_down_time_c(model, i):
    if i == T1:
        return (d5_down_before + d5_min_down_time) * (-u5_start + m.u5[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u5[i - 1])
        for j in range(T1, i):
            uj.append(m.u5[j])
        return ((sum(uj) - 2 + d5_min_down_time) * (-m.u5[i - 1] + m.u5[i])) <= 0
    else:
        return ((m.u5[i - 1] + m.u5[i - 2] + m.u5[i - 3] - 3 + d5_min_down_time) * (-m.u5[i - 1] + m.u5[i])) <= 0


def d6_min_down_time_c(model, i):
    if i == T1:
        return (d6_down_before + d6_min_down_time) * (-u6_start + m.u6[i]) <= 0
    elif i <= T1 + 2:
        uj = []
        uj.append(m.u6[i - 1])
        for j in range(T1, i):
            uj.append(m.u6[j])
        return ((sum(uj) - 2 + d6_min_down_time) * (-m.u6[i - 1] + m.u6[i])) <= 0
    else:
        return ((m.u6[i - 1] + m.u6[i - 2] + m.u6[i - 3] - 3 + d6_min_down_time) * (-m.u6[i - 1] + m.u6[i])) <= 0


def d1_start_up_cost(model, i):
    if i == T1:
        return m.suc1[i] >= d1_startup_cost * (m.u1[i] - u1_start)
    else:
        return m.suc1[i] >= d1_startup_cost * (m.u1[i] - m.u1[i - 1])


def d2_start_up_cost(model, i):
    if i == T1:
        return m.suc2[i] >= d2_startup_cost * (m.u2[i] - u2_start)
    else:
        return m.suc2[i] >= d2_startup_cost * (m.u2[i] - m.u2[i - 1])


def d3_start_up_cost(model, i):
    if i == T1:
        return m.suc3[i] >= d3_startup_cost * (m.u3[i] - u3_start)
    else:
        return m.suc3[i] >= d3_startup_cost * (m.u3[i] - m.u3[i - 1])


def d4_start_up_cost(model, i):
    if i == T1:
        return m.suc4[i] >= d4_startup_cost * (m.u4[i] - u4_start)
    else:
        return m.suc4[i] >= d4_startup_cost * (m.u4[i] - m.u4[i - 1])


def d5_start_up_cost(model, i):
    if i == T1:
        return m.suc5[i] >= d5_startup_cost * (m.u5[i] - u5_start)
    else:
        return m.suc5[i] >= d5_startup_cost * (m.u5[i] - m.u5[i - 1])


def d6_start_up_cost(model, i):
    if i == T1:
        return m.suc6[i] >= d6_startup_cost * (m.u6[i] - u6_start)
    else:
        return m.suc6[i] >= d6_startup_cost * (m.u6[i] - m.u6[i - 1])


def d1_shut_down_cost(model, i):
    if i == T1:
        return m.sdc1[i] >= d1_shutdown_cost * (u1_start - m.u1[i])
    else:
        return m.sdc1[i] >= d1_shutdown_cost * (m.u1[i - 1] - m.u1[i])


def d2_shut_down_cost(model, i):
    if i == T1:
        return m.sdc2[i] >= d2_shutdown_cost * (u2_start - m.u2[i])
    else:
        return m.sdc2[i] >= d2_shutdown_cost * (m.u2[i - 1] - m.u2[i])


def d3_shut_down_cost(model, i):
    if i == T1:
        return m.sdc3[i] >= d3_shutdown_cost * (u3_start - m.u3[i])
    else:
        return m.sdc3[i] >= d3_shutdown_cost * (m.u3[i - 1] - m.u3[i])


def d4_shut_down_cost(model, i):
    if i == T1:
        return m.sdc4[i] >= d4_shutdown_cost * (u4_start - m.u4[i])
    else:
        return m.sdc4[i] >= d4_shutdown_cost * (m.u4[i - 1] - m.u4[i])


def d5_shut_down_cost(model, i):
    if i == T1:
        return m.sdc5[i] >= d5_shutdown_cost * (u5_start - m.u5[i])
    else:
        return m.sdc5[i] >= d5_shutdown_cost * (m.u5[i - 1] - m.u5[i])


def d6_shut_down_cost(model, i):
    if i == T1:
        return m.sdc6[i] >= d6_shutdown_cost * (u6_start - m.u6[i])
    else:
        return m.sdc6[i] >= d6_shutdown_cost * (m.u6[i - 1] - m.u6[i])


def soc1_ctrl(model, i):
    if i == T1:
        return m.soc1[i] == soc1_before - 100*m.bat1_dch[i]/(Eff_Bat*BatteryCapacity) + Eff_Bat*100*m.bat1_ch[i]/BatteryCapacity
    else:
        return m.soc1[i] == m.soc1[i-1] - 100*m.bat1_dch[i]/(Eff_Bat*BatteryCapacity) + Eff_Bat*100*m.bat1_ch[i]/BatteryCapacity

def soc2_ctrl(model, i):
    if i == T1:
        return m.soc2[i] == soc2_before - 100*m.bat2_dch[i]/(Eff_Bat*BatteryCapacity) + Eff_Bat*100*m.bat2_ch[i]/BatteryCapacity
    else:
        return m.soc2[i] == m.soc2[i-1] - 100*m.bat2_dch[i]/(Eff_Bat*BatteryCapacity) + Eff_Bat*100*m.bat2_ch[i]/BatteryCapacity


def ess1_availability (model, i):
    return m.bat1_dch[i] + m.bat1_ch[i] == (m.bat1_dch[i] + m.bat1_ch[i]) * ess1_availability_state

def ess2_availability (model, i):
    return m.bat2_dch[i] + m.bat2_ch[i] == (m.bat2_dch[i] + m.bat2_ch[i]) * ess2_availability_state

# СНЭ не должна заряжаться и разряжаться одновременно
def ch_x_dch1 (model, i):
    return m.bat1_ch[i] * m.bat1_dch[i] == 0

def ch_x_dch2 (model, i):
    return m.bat2_ch[i] * m.bat2_dch[i] == 0

def ess12(model, i):
    return m.bat1_dch[i] - m.bat2_ch[i] - m.bat1_dch[i] == 0

def ess21(model, i):
    return m.bat2_dch[i] - m.bat1_ch[i] - m.bat2_dch[i] == 0

def cycle1(model, i):
    return m.soc1[T1+T2-T1-1] >= soc1_after

def cycle2(model, i):
    return m.soc2[T1+T2-T1-1] >= soc2_after

#синхронная работа инверторов СНЭ
def as_one1(model, i):
    return m.bat1_dch[i] == m.bat2_dch[i]

def as_one2(model, i):
    return m.bat1_ch[i] == m.bat2_ch[i]

# если СНЭ заряжена не на 100%, не ограничиваем инверторы СЭС
def curtailment_control1 (model, i):
    return m.PV1[i] + m.PV2[i] + m.PV3[i] + m.PV4[i] + m.PV5[i] + m.PV6[i] + m.PV7[i] - PV[i] - m.bat1_ch[i]/ESS_inv  <= 0

def curtailment_control2 (model, i):
    return m.PV1[i] + m.PV2[i] + m.PV3[i] + m.PV4[i] + m.PV5[i] + m.PV6[i] + m.PV7[i] - PV[i] - m.bat2_ch[i]/ESS_inv  <= 0

def d1_availability (model, i):
    return m.u1[i] == m.u1[i] * d1_availability_state

def d2_availability (model, i):
    return m.u2[i] == m.u2[i] * d2_availability_state

def d3_availability (model, i):
    return m.u3[i] == m.u3[i] * d3_availability_state

def d4_availability (model, i):
    return m.u4[i] == m.u4[i] * d4_availability_state

def d5_availability (model, i):
    return m.u5[i] == m.u5[i] * d5_availability_state

def d6_availability (model, i):
    return m.u6[i] == m.u6[i] * d6_availability_state


def unit_commitment():
    m.N = pyo.Set(initialize=N)
    m.T = pyo.Set(initialize=T)

    m.x1 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU1_pmax))
    m.u1 = pyo.Var(m.T, domain=pyo.Binary)

    m.x2 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU2_pmax))
    m.u2 = pyo.Var(m.T, domain=pyo.Binary)

    m.x3 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU3_pmax))
    m.u3 = pyo.Var(m.T, domain=pyo.Binary)

    m.x4 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU4_pmax))
    m.u4 = pyo.Var(m.T, domain=pyo.Binary)

    m.x5 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU5_pmax))
    m.u5 = pyo.Var(m.T, domain=pyo.Binary)

    m.x6 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, DGU6_pmax))
    m.u6 = pyo.Var(m.T, domain=pyo.Binary)

    m.bat1_dch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, ESS_inv))
    m.bat1_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, ESS_inv))

    m.bat2_dch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, ESS_inv))
    m.bat2_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, ESS_inv))

    m.soc1 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(SOCmin, SOCmax))
    m.soc2 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(SOCmin, SOCmax))

    m.suc1 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d1_startup_cost))
    m.suc2 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d2_startup_cost))
    m.suc3 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d3_startup_cost))
    m.suc4 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d4_startup_cost))
    m.suc5 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d5_startup_cost))
    m.suc6 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d6_startup_cost))

    m.sdc1 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d1_shutdown_cost))
    m.sdc2 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d2_shutdown_cost))
    m.sdc3 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d3_shutdown_cost))
    m.sdc4 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d4_shutdown_cost))
    m.sdc5 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d5_shutdown_cost))
    m.sdc6 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, d6_shutdown_cost))

    m.PV1 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv1))
    m.PV2 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv2))
    m.PV3 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv3))
    m.PV4 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv4))
    m.PV5 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv5))
    m.PV6 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv6))
    m.PV7 = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, PV_inv7))

    m.lb1 = pyo.Constraint(m.T, rule=lambda m, t: DGU1_pmin * m.u1[t] <= m.x1[t])
    m.ub1 = pyo.Constraint(m.T, rule=lambda m, t: DGU1_pmax * m.u1[t] >= m.x1[t])
    m.lb2 = pyo.Constraint(m.T, rule=lambda m, t: DGU2_pmin * m.u2[t] <= m.x2[t])
    m.ub2 = pyo.Constraint(m.T, rule=lambda m, t: DGU2_pmax * m.u2[t] >= m.x2[t])
    m.lb3 = pyo.Constraint(m.T, rule=lambda m, t: DGU3_pmin * m.u3[t] <= m.x3[t])
    m.ub3 = pyo.Constraint(m.T, rule=lambda m, t: DGU3_pmax * m.u3[t] >= m.x3[t])
    m.lb4 = pyo.Constraint(m.T, rule=lambda m, t: DGU4_pmin * m.u4[t] <= m.x4[t])
    m.ub4 = pyo.Constraint(m.T, rule=lambda m, t: DGU4_pmax * m.u4[t] >= m.x4[t])
    m.lb5 = pyo.Constraint(m.T, rule=lambda m, t: DGU5_pmin * m.u5[t] <= m.x5[t])
    m.ub5 = pyo.Constraint(m.T, rule=lambda m, t: DGU5_pmax * m.u5[t] >= m.x5[t])
    m.lb6 = pyo.Constraint(m.T, rule=lambda m, t: DGU6_pmin * m.u6[t] <= m.x6[t])
    m.ub6 = pyo.Constraint(m.T, rule=lambda m, t: DGU6_pmax * m.u6[t] >= m.x6[t])

    m.pv1 = pyo.Constraint(m.T, rule=lambda m, t: m.PV1[t] <= PV[t] / 7)
    m.pv2 = pyo.Constraint(m.T, rule=lambda m, t: m.PV2[t] <= PV[t] / 7)
    m.pv3 = pyo.Constraint(m.T, rule=lambda m, t: m.PV3[t] <= PV[t] / 7)
    m.pv4 = pyo.Constraint(m.T, rule=lambda m, t: m.PV4[t] <= PV[t] / 7)
    m.pv5 = pyo.Constraint(m.T, rule=lambda m, t: m.PV5[t] <= PV[t] / 7)
    m.pv6 = pyo.Constraint(m.T, rule=lambda m, t: m.PV6[t] <= PV[t] / 7)
    m.pv7 = pyo.Constraint(m.T, rule=lambda m, t: m.PV7[t] <= PV[t] / 7)

    m.pv1_curtailment = pyo.Constraint(m.T, rule=curtailment_control1)
    m.pv2_curtailment = pyo.Constraint(m.T, rule=curtailment_control2)

    m.min_up_time_d1 = pyo.Constraint(m.T, rule=d1_min_up_time_c)
    m.min_up_time_d2 = pyo.Constraint(m.T, rule=d2_min_up_time_c)
    m.min_up_time_d3 = pyo.Constraint(m.T, rule=d3_min_up_time_c)
    m.min_up_time_d4 = pyo.Constraint(m.T, rule=d4_min_up_time_c)
    m.min_up_time_d5 = pyo.Constraint(m.T, rule=d5_min_up_time_c)
    m.min_up_time_d6 = pyo.Constraint(m.T, rule=d6_min_up_time_c)

    m.min_down_time_d1 = pyo.Constraint(m.T, rule=d1_min_down_time_c)
    m.min_down_time_d2 = pyo.Constraint(m.T, rule=d2_min_down_time_c)
    m.min_down_time_d3 = pyo.Constraint(m.T, rule=d3_min_down_time_c)
    m.min_down_time_d4 = pyo.Constraint(m.T, rule=d4_min_down_time_c)
    m.min_down_time_d5 = pyo.Constraint(m.T, rule=d5_min_down_time_c)
    m.min_down_time_d6 = pyo.Constraint(m.T, rule=d6_min_down_time_c)

    m.su1 = pyo.Constraint(m.T, rule=d1_start_up_cost)
    m.su2 = pyo.Constraint(m.T, rule=d2_start_up_cost)
    m.su3 = pyo.Constraint(m.T, rule=d3_start_up_cost)
    m.su4 = pyo.Constraint(m.T, rule=d4_start_up_cost)
    m.su5 = pyo.Constraint(m.T, rule=d5_start_up_cost)
    m.su6 = pyo.Constraint(m.T, rule=d6_start_up_cost)

    m.sd1 = pyo.Constraint(m.T, rule=d1_shut_down_cost)
    m.sd2 = pyo.Constraint(m.T, rule=d2_shut_down_cost)
    m.sd3 = pyo.Constraint(m.T, rule=d3_shut_down_cost)
    m.sd4 = pyo.Constraint(m.T, rule=d4_shut_down_cost)
    m.sd5 = pyo.Constraint(m.T, rule=d5_shut_down_cost)
    m.sd6 = pyo.Constraint(m.T, rule=d6_shut_down_cost)

    # Ограничение - доступность ДГУ для работы
    m.d1_ava = pyo.Constraint(m.T, rule=d1_availability)
    m.d2_ava = pyo.Constraint(m.T, rule=d2_availability)
    m.d3_ava = pyo.Constraint(m.T, rule=d3_availability)
    m.d4_ava = pyo.Constraint(m.T, rule=d4_availability)
    m.d5_ava = pyo.Constraint(m.T, rule=d5_availability)
    m.d6_ava = pyo.Constraint(m.T, rule=d6_availability)

    m.soc1_ctrl = pyo.Constraint(m.T, rule=soc1_ctrl)
    m.soc2_ctrl = pyo.Constraint(m.T, rule=soc2_ctrl)
    m.chdch1 = pyo.Constraint(m.T, rule=ch_x_dch1)
    m.chdch2 = pyo.Constraint(m.T, rule=ch_x_dch2)
    # m.ess12 = pyo.Constraint(m.T, rule=ess12)
    # m.ess21 = pyo.Constraint(m.T, rule=ess21)
    m.ess1_ava = pyo.Constraint(m.T, rule=ess1_availability)
    m.ess2_ava = pyo.Constraint(m.T, rule=ess2_availability)
    # m.d1_cstr1 = pyo.Constraint(m.T, rule=d1_cstr1)
    # m.curt1 = pyo.Constraint(m.T, rule=curtailment_control1)
    # m.ess1_cyc = pyo.Constraint(m.T, rule=cycle1)
    # m.ess2_cyc = pyo.Constraint(m.T, rule=cycle2)

    m.as_one1 = pyo.Constraint(m.T, rule=as_one1)
    m.as_one2 = pyo.Constraint(m.T, rule=as_one2)

    # Ограничение баланса мощности
    m.demand = pyo.Constraint(m.T, rule=balance)

    # целевая функция
    m.cost = pyo.Objective(expr=sum(Fuel_price * ((Kb1 * m.x1[t] + Bb1) + \
                                                  (Kb2 * m.x2[t] + Bb2) + \
                                                  (Kb3 * m.x3[t] + Bb3) + \
                                                  (Kb4 * m.x4[t] + Bb4) + \
                                                  (Kb5 + m.x5[t] + Bb5) + \
                                                  (Kb6 + m.x6[t] + Bb6)) \
                                    for t in m.T), sense=pyo.minimize)

    return m


m = unit_commitment()

opt = pyo.SolverFactory('couenne')
opt.set_executable('couenne.exe', validate=False)
# Обращаемся к алгоритмическому решателю задач типа MINLP
# results = opt.solve(m, logfile=r'couenne_logg.log', tee=True, timelimit=6000, keepfiles=True)
# results = opt.solve(m)
results = opt.solve(m, load_solutions=False)
if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal):
    print('Solve OK')

    m.solutions.load_from(results)
    # results.write()

    # Собираем результаты оптимизации в читаемый датафрейм
    result_df = pd.DataFrame()
    result_df['D1, kW'] = m.x1.get_values().values()
    result_df['D2, kW'] = m.x2.get_values().values()
    result_df['D3, kW'] = m.x3.get_values().values()
    result_df['D4, kW'] = m.x4.get_values().values()
    result_df['D5, kW'] = m.x5.get_values().values()
    result_df['D6, kW'] = m.x6.get_values().values()
    result_df['CH1, kW'] = m.bat1_ch.get_values().values()
    result_df['dCH1, kW'] = m.bat1_dch.get_values().values()
    result_df['CH2, kW'] = m.bat2_ch.get_values().values()
    result_df['dCH2, kW'] = m.bat2_dch.get_values().values()
    result_df['SOC1, %'] = m.soc1.get_values().values()
    result_df['SOC2, %'] = m.soc2.get_values().values()
    result_df['PV1, kW'] = m.PV1.get_values().values()
    result_df['PV2, kW'] = m.PV2.get_values().values()
    result_df['PV3, kW'] = m.PV3.get_values().values()
    result_df['PV4, kW'] = m.PV4.get_values().values()
    result_df['PV5, kW'] = m.PV5.get_values().values()
    result_df['PV6, kW'] = m.PV6.get_values().values()
    result_df['PV7, kW'] = m.PV7.get_values().values()

    result_df.replace(to_replace=[None], value=np.nan, inplace=True)

    result_df['Diesel, kW'] = result_df['D1, kW'] + result_df['D2, kW'] + result_df['D3, kW'] + result_df['D4, kW'] + \
                              result_df['D5, kW'] + result_df['D6, kW']
    result_df['ESS, kW'] = -result_df['CH1, kW'] + result_df['dCH1, kW'] - result_df['CH2, kW'] + result_df['dCH2, kW']
    result_df['PV, kW'] = result_df['PV1, kW'] + result_df['PV2, kW'] + result_df['PV3, kW'] + result_df['PV4, kW'] + \
                          result_df['PV5, kW'] + result_df['PV6, kW'] + result_df['PV7, kW']
    result_df['Load, kW'] = Load[T1:T2]
else:
    print('Solve failed')

    #
    # ЗДЕСЬ НУЖНО ВСТАВИТЬ "БАЗОВЫЙ ВАРИАНТ" РАСЧЁТЫ РЕЖИМА РАБОТЫ СДК
    #

    # Собираем результаты оптимизации в читаемый датафрейм
    result_df = pd.DataFrame()
    result_df['D1, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['D2, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['D3, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['D4, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['D5, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['D6, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['CH1, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['dCH1, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['CH2, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['dCH2, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['SOC1, %'] = soc1_before
    result_df['SOC2, %'] = soc2_before
    result_df['PV1, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV2, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV3, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV4, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV5, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV6, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_df['PV7, kW'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # result_df.replace(to_replace=[None], value=np.nan, inplace=True)

    result_df['Diesel, kW'] = result_df['D1, kW'] + result_df['D2, kW'] + result_df['D3, kW'] + result_df['D4, kW'] + \
                              result_df['D5, kW'] + result_df['D6, kW']
    result_df['ESS, kW'] = -result_df['CH1, kW'] + result_df['dCH1, kW'] - result_df['CH2, kW'] + result_df['dCH2, kW']
    result_df['PV, kW'] = result_df['PV1, kW'] + result_df['PV2, kW'] + result_df['PV3, kW'] + result_df['PV4, kW'] + \
                          result_df['PV5, kW'] + result_df['PV6, kW'] + result_df['PV7, kW']
    result_df['Load, kW'] = Load[T1:T2]

result_df.to_csv('result.csv')


def plot_summary(df):
    fig, ax = plt.subplots(2, 1, figsize=(18, 24))

    PV_power_df = df[['PV1, kW', 'PV2, kW', 'PV3, kW', 'PV4, kW', 'PV5, kW', 'PV6, kW', 'PV7, kW']]
    PV_power = PV_power_df.sum(axis=1)

    ax[0].set_ylabel('Мощность, кВт', fontsize=15)
    ax[0].set_xlabel('Номер часа', fontsize=15)
    ax[0].bar(df.index, PV_power, label='СЭС', edgecolor="black", width=0.75, hatch='//', color='orange')
    ax[0].bar(df.index, -df['CH1, kW'], width=0.75, edgecolor='black', hatch='o', color='slateblue')
    ax[0].bar(df.index, -df['CH2, kW'], bottom=-df['CH1, kW'], width=0.75, edgecolor='black', hatch='o',
              color='darkslateblue')
    ax[0].bar(df.index, df['dCH1, kW'], bottom=df['D3, kW'] + df['D4, kW'] + df['D1, kW'] + df['D2, kW'] + PV_power,
              label='СНЭ 1', width=0.75, edgecolor='black', hatch='o', color='slateblue')
    ax[0].bar(df.index, df['dCH2, kW'],
              bottom=df['D3, kW'] + df['D4, kW'] + df['D1, kW'] + df['D2, kW'] + df['dCH1, kW'] + PV_power,
              label='СНЭ 2', width=0.75, edgecolor='black', hatch='o', color='darkslateblue')
    ax[0].bar(df.index, df['D1, kW'], label='ДГУ 1', bottom=PV_power + df['D3, kW'] + df['D4, kW'], edgecolor="black",
              align='center', width=0.75, hatch="//", color='royalblue')
    ax[0].bar(df.index, df['D2, kW'], label='ДГУ 2', bottom=PV_power + df['D3, kW'] + df['D4, kW'] + df['D1, kW'],
              edgecolor="black", align='center', width=0.75, hatch="//", color='dimgray')
    ax[0].bar(df.index, df['D3, kW'], label='ДГУ 3', bottom=PV_power, edgecolor="black", align='center', width=0.75,
              hatch="//", color='teal')
    ax[0].bar(df.index, df['D4, kW'], label='ДГУ 4', bottom=PV_power + df['D3, kW'], edgecolor="black", align='center',
              width=0.75, hatch="//", color='deepskyblue')
    ax[0].plot(df.index, df['Load, kW'], color='red', label='Нагрузка', linewidth=4, marker='o',
               markeredgecolor='black', markersize=10)
    ax[0].legend(fontsize=15)
    ax[0].grid()

    ax[1].set_ylabel('Уровень заряда АКБ, %', fontsize=15)
    ax[1].set_xlabel('Номер часа', fontsize=15)
    ax[1].set_ylim([SOCmin * 0.9, SOCmax * 1.1])
    ax[1].plot(df.index, df['SOC1, %'], '--', color='darkred', label='SOC 1', linewidth=3, marker='o',
               markeredgecolor='black', markersize=10)
    ax[1].plot(df.index, df['SOC2, %'], '--', color='darkred', label='SOC 2', linewidth=3, marker='s',
               markeredgecolor='black', markersize=10)
    ax[1].legend(fontsize=15)
    ax[1].grid()

    plt.show()

    return


plot_summary(result_df)