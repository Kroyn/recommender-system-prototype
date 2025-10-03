import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit


# Дані з таблиці 3.1
x_data = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
y_data = np.array([4-1, 6-1, 6-1, 7-1, 7-1, 8-1, 8-1, 9-1, 9-1, 10-1, 10-1, 11-1, 11-1])


# Лінійна модель
X_linear = sm.add_constant(x_data)
model_linear = sm.OLS(y_data, X_linear).fit()
print('Лінійна модель R^2:', model_linear.rsquared)
b0_linear, b1_linear = model_linear.params
print(f'Лінійна: y = {b0_linear:.4f} + {b1_linear:.4f}x')


# Експоненціальна модель y = b0 * exp(b1*x)
def exp_func(x, b0, b1):
    return b0 * np.exp(b1 * x)
popt_exp, _ = curve_fit(exp_func, x_data, y_data, p0=[1, 0.1])
y_pred_exp = exp_func(x_data, *popt_exp)
ss_res_exp = np.sum((y_data - y_pred_exp)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2_exp = 1 - (ss_res_exp / ss_tot)
print(f'Експоненціальна: y = {popt_exp[0]:.4f} * exp({popt_exp[1]:.4f}x), R^2 = {r2_exp:.4f}')


# Степенева модель y = b0 * x^b1
def power_func(x, b0, b1):
    return b0 * np.power(x, b1)
popt_power, _ = curve_fit(power_func, x_data, y_data, p0=[1, 0.1])
y_pred_power = power_func(x_data, *popt_power)
ss_res_power = np.sum((y_data - y_pred_power)**2)
r2_power = 1 - (ss_res_power / ss_tot)
print(f'Степенева: y = {popt_power[0]:.4f} * x^{popt_power[1]:.4f}, R^2 = {r2_power:.4f}')


# Логарифмічна модель y = b0 + b1 * ln(x)
X_log = sm.add_constant(np.log(x_data))
model_log = sm.OLS(y_data, X_log).fit()
print('Логарифмічна модель R^2:', model_log.rsquared)
b0_log, b1_log = model_log.params
print(f'Логарифмічна: y = {b0_log:.4f} + {b1_log:.4f} * ln(x)')
