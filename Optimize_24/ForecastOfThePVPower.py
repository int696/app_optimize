# Модуль прогнозирования солнечного излучения на ближайшие 24 часа
# В основе функционирования модуля лежат два инструмента:
# 1. Сервис Нарынбаева А.Ф. по прогнозированию солнечного излучения на основе метеопрогноза. С
#    использованием его API в разделе 3 осуществляется запрос данных;
# 2. Библиотека PVLIB - с её помощью осуществляется пересчёт СИ с горизонтальной поверхности на
#    наклонную, моделирование ВАХ батареи ФЭМ и расчёт выработки электроэнергии со стороны
#    переменного тока инвертора.
#
import Weather.weather
import pandas as pd
import pvlib
import datetime
from Connected.connection_db import add_user


def get_forecast_pv_power(connect):
    cursor = connect.cursor()
    # Получение информации из таблицы parameters_of_SDC
    cursor.execute("SELECT * FROM Параметры_СДК")
    parameters_of_SDC = cursor.fetchone()
    print('OK_1:', parameters_of_SDC)
    # Получение информации из таблицы parameters_of_pv_modules
    cursor.execute("SELECT * FROM Параметры_модель_СЭС")
    parameters_of_pv_modules = cursor.fetchone()
    print('OK_2:', parameters_of_pv_modules)
    # Получение информации из таблицы parameters_inv
    cursor.execute("SELECT * FROM Параметры_инвертора")
    parameters_inv = cursor.fetchone()
    print('OK_3:', parameters_inv)
    cursor.close()

    ### 1.2 Меcторасположение СЭС ########################################################################

    # Широта расположения СДК
    #latitude = 55.72
    latitude = parameters_of_SDC['latitude']

    # Долгота расположения СДК
    #longitude = 37.55
    longitude = parameters_of_SDC['longitude']

    # Высота расположения СДК над уровнем моря, м
    #altitude = 127
    altitude = parameters_of_SDC['altitude']

    # Наименование часового пояса, в котором располагается СДК
    #TimeZone = 'Europe/Moscow'
    TimeZone = parameters_of_SDC['timezone']

    # Название СЭС
    #NameA = 'Moscow'
    NameA = parameters_of_SDC['name_of_SPS']

    # Угол наклона поверхности ФЭП, градусы
    #surface_tilt = 55
    surface_tilt = parameters_of_pv_modules['surface_tilt']
    # Азимут поверхности ФЭП, градусы
    #surface_azimuth = 180
    surface_azimuth = parameters_of_pv_modules['surface_azimuth']
    # Количество ФЭМ в цепочке
    #M = 8
    M = parameters_of_pv_modules['number_of_pvmodules_in_circuit']
    # Количество цепочек, подключённых к одному инвертору
    #N = 1
    N = parameters_of_pv_modules['number_of_circuits_con_to_inverter']
    # Количество инверторов на СЭС
    #Ni = 1
    Ni = parameters_of_pv_modules['number_of_inverters_in_SPS']

    # Параметры используемого ФЭМ при стандартных условиях
    # (определить можно с помощью функции pvlib.ivtools.sdm.fit_cec_sam)
    # Коэффициент идеальности диода, о.е.
    #a_ref = 2.8727862399469153
    a_ref = parameters_of_pv_modules['a_ref']
    # Фототок, А
    #I_L_ref = 10.605174110772369
    I_L_ref = parameters_of_pv_modules['I_L_ref']
    # "Темновой" ток, А
    #I_o_ref = 9.748765307697816e-08
    I_o_ref = parameters_of_pv_modules['I_o_ref']
    # Шунтирующее сопротивление, Ом
    #R_sh_ref = 702.2821810615955
    R_sh_ref = parameters_of_pv_modules['R_sh_ref']
    # Последовательное сопротивление, Ом
    #R_s = 0.01152964797494233
    R_s = parameters_of_pv_modules['R_s']
    # Температурный коэффициент тока корткого замыкания, A/C
    #alpha_sc = 0.055
    alpha_sc = parameters_of_pv_modules['alpha_sc']

    # Параметры, необходимые для расчёта температуры поверхности ячеек
    #a = -3.47
    a = parameters_of_pv_modules['a']
    #b = -0.0594
    b = parameters_of_pv_modules['b']
    #deltaT = 3
    delta_T = parameters_of_pv_modules['delta_T']

    # Эффективная иррадиация, о.е.
    #eff_irr = 0.97
    eff_irr = parameters_of_pv_modules['eff_irr']

    # параметры инвертора
    #pdc0 = 12000 # мощность инвертора, Вт
    pdc0 = parameters_inv['pdc0']
    #eta_inv_nom = 0.979 # КПД инвертора при номинальной загрузке
    eta_inv_nom = parameters_inv['eta_inv_nom']
    #eta_inv_ref = 0.982 # КПД инвертора при оптимальной загрузке
    eta_inv_ref = parameters_inv['eta_inv_ref']


    # Задаём параметра месторасположения СЭС
    A = pvlib.location.Location(latitude,
                                longitude,
                                tz=TimeZone,
                                altitude=altitude,
                                name=NameA)
    # Загрузка метеорологической информации в TMY-формате
    # данные должны быть заранее скачаны, например, с PVGIS, и адаптированы к месту размещения СЭС.
    # имя фала - TMY_.csv
    # разделитель - ';'
    tmy = pd.read_csv('TMY_.csv', sep=';')
    tmy = tmy._append(tmy.iloc[0:24], ignore_index = True) # Копирование первых суток года в конец,
    # чтобы можно было работать с 31 декабря



    ########################################################################################################################
    # 2 Определение моментов времени, начиная от текущего часа и на 24 часа вперёд
    ########################################################################################################################
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    beginning_of_year = datetime.datetime(current_year, 1, 1)
    tt = (now - beginning_of_year).total_seconds() // 3600
    if current_year % 4 == 0 and tt>=1416: # Если год - високосный, то 29 февраля приравнивается к 28 числу
        tt = tt-24
    StartTime = datetime.datetime(current_year, current_month, current_day, current_hour)
    EndTime = StartTime + datetime.timedelta(hours=23)
    times = pd.date_range(start=StartTime,
                          end=EndTime,
                          freq='1h')
    times_loc = times.tz_localize(A.pytz)
    tmy24 = tmy.iloc[int(tt)-1:int(tt)+24-1]
    tmy24 = tmy24.assign(time = times_loc)
    tmy24.set_index('time', inplace=True) # метки времени устанавливаются в dataframe в качестве индексов



    ########################################################################################################################
    ### 3 Получение метеопрогноза с помощью сервиса Нарынбаева А.Ф.
    ########################################################################################################################
    test = Weather.weather.WeatherForecast()
    test.set_param(station=1, var_station='mpei', var_nwp_provider='icon')
    test.get_json()
    i = 0
    if test.status_code == 200:
        # Если обращение к сервису завершилось удачно, то часть столбцов в предварительно загруженном dataframe
        # заменяется на необходимые значения
        for i in range(24):
            direct_radiation = test.get_param_weather(i, 'direct_radiation').values
            direct_normal_irradiance = test.get_param_weather(i, 'direct_normal_irradiance').values
            terrestrial_radiation = test.get_param_weather(i, 'terrestrial_radiation').values
            diffuse_radiation = test.get_param_weather(i, 'diffuse_radiation').values
            surface_pressure = test.get_param_weather(i, 'surface_pressure').values
            temperature_2m = test.get_param_weather(i, 'temperature_2m').values
            windspeed_10m = test.get_param_weather(i, 'windspeed_10m').values
            # Номера столбцов в dataframe:
            # time - индекс
            # HOY 	                0
            # month                 1
            # day 	                2
            # hour 	                3
            # temp_air 	            4
            # atmospheric_pressure 	5
            # ghi 	                6
            # dni 	                7
            # dhi 	                8
            # wind_speed 	        9
            # albedo                10
            tmy24.iloc[i, 4] = temperature_2m
            tmy24.iloc[i, 5] = surface_pressure
            tmy24.iloc[i, 6] = terrestrial_radiation  # GHI - Global Horizontal Irradiance
            tmy24.iloc[i, 7] = direct_normal_irradiance  # DNI - Diffused Normal Irradiance
            tmy24.iloc[i, 8] = diffuse_radiation  # DHI - Diffused Horizontal Irradiance
            tmy24.iloc[i, 9] = windspeed_10m



    ########################################################################################################################
    ### 4 Расчёт солнечного излучения падающего в плоскости ФЭМ
    ########################################################################################################################
    DNI_extra = pvlib.irradiance.get_extra_radiation(times_loc, method='spencer')
    SPA = pvlib.solarposition.spa_python(times_loc, A.latitude, A.longitude, A.altitude)
    airmass_relative = pvlib.atmosphere.get_relative_airmass(SPA['zenith'], model='simple')
    DiffuseSky = pvlib.irradiance.perez(surface_tilt,
                                              surface_azimuth,
                                              tmy24['dhi'],
                                              tmy24['dni'],
                                              DNI_extra,
                                              SPA['apparent_zenith'],
                                              SPA['azimuth'],
                                              airmass_relative,
                                              model='allsitescomposite1990',
                                              return_components=False)
    DiffuseGround = pvlib.irradiance.get_ground_diffuse(surface_tilt,
                                                        tmy24['ghi'],
                                                        albedo=tmy24['albedo'],
                                                        surface_type=None)
    AOI = pvlib.irradiance.aoi(surface_tilt,
                               surface_azimuth,
                               SPA['zenith'],
                               SPA['azimuth'])
    GTI = pvlib.irradiance.poa_components(AOI,
                                        tmy24['dni'],
                                        DiffuseSky,
                                        DiffuseGround)
    GTI = GTI.fillna(0)



    ########################################################################################################################
    ### 5 Моделирование выходных показателей единичного ФЭМ, батареи ФЭМ, единичного инвертора и всей СЭС в целом
    ########################################################################################################################
    effective_irradiance = GTI['poa_global'] * eff_irr
    Cells_temperature = pvlib.temperature.sapm_cell(
        GTI['poa_global'].values,
        tmy24['temp_air'].values,
        tmy24['wind_speed'].values,
        a,
        b,
        delta_T,
        irrad_ref=1000.0)
    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance,
        Cells_temperature,
        alpha_sc,
        a_ref,
        I_L_ref,
        I_o_ref,
        R_sh_ref,
        R_s,
        EgRef=1.121,
        dEgdT= -0.0002677,
        irrad_ref=1000,
        temp_ref=25)
    DCout = pvlib.pvsystem.singlediode(
        photocurrent,
        saturation_current,
        resistance_series,
        resistance_shunt,
        nNsVth,
        method='newton')
    VDC_array = DCout['v_mp'] * M
    IDC_array = DCout['i_mp'] * N
    PDC_array = DCout['p_mp'] * M * N
    ACout = pvlib.inverter.pvwatts(PDC_array, pdc0, eta_inv_nom=eta_inv_nom, eta_inv_ref=eta_inv_ref)
    Npv_forecast = ACout * Ni
    return Npv_forecast


if __name__ == '__main__':
    user_bd = add_user()
    get_forecast_pv_power(user_bd)

