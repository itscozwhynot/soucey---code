import iv
import Cubic_spline as CS
from suns_voc_parse import SUNS_VOC_PARSE
from loana_parse import LOANA_PARSE
#####################################################################################################
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, splev
from scipy.stats import linregress
####################################################################################################

import PySpice.Logging.Logging as Logging
#logger = Logging.setup_logging()

####################################################################################################

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.NgSpice.Simulation import NgSpiceSubprocessCircuitSimulator
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

######################################################################################################
#HELPFUL FUNCTIONS###########
######################################################################################################

#splice x,y data based on an x range
def splice(data,minval = None,maxval = None):
    if (minval is None):
        minval = min(data[0])
    
    if (maxval is None):
        maxval = max(data[0])
    
    xy_ = np.column_stack((data))

    xy_ = np.delete(xy_, np.argwhere( (xy_[:,0] <= minval)), axis = 0)
    xy_ = np.delete(xy_, np.argwhere( (xy_[:,0] >= maxval)), axis = 0)
    #x,y,z = xy_[:,0], xy_[:,1], xy_[:,2]

    xy_ = np.swapaxes(xy_,0,1)
    return xy_

#function to sort data in increasing order for interpolate
def array_sort(x,y):
    data = np.column_stack((x, y))
    data_sorted = data[data[:,0].argsort()]
    x,y = data_sorted[:,0], data_sorted[:,1]
    return x,y

#find the R_squared value of the fit (normalised)
def R_squared(x,y,y_fit):
    '''
    x is the independent variable
    y is the dependent variable
    y_fit is the fitted dependent variable
    '''
    residuals = (y-y_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(((y-np.mean(y)))**2)
    r_squared = 1- (ss_res/ss_tot)
    return float(r_squared)
######################################################################################################
#SIMULATION FUNCTIONS###########
######################################################################################################

def circuit_out(circuit, vlts, temp):

    simulator = circuit.simulator(temperature=temp, nominal_temperature=temp)
    analysis = simulator.dc(Vinput = slice(vlts[0], vlts[-1], .001))

    # get current response [A]
    I = np.array(analysis.Vinput)
    V = np.array(analysis.sweep)

    spl = splrep(V, I)
    _I = splev(vlts, spl)

    return np.array(_I)

def circuit(params):
     '''
         build equivalent circuit
     '''

     # initialise circuit
     circuit = Circuit('test_circuit')

     #capacitive current
     #equation = '-' + str(params['C']) + '*(1 + V(rs_in,gnd)/0.0258)'
     #circuit.B('cap', circuit.gnd, 'rs_out', current_expression = equation)

     # initialise voltage source
     circuit.V('input', 'rs_in', circuit.gnd, 1@u_V)

     # initialise series resistance
     circuit.R('rs', 'rs_in', 'rs_out', params['rs'])

     # initialise parallel resistance
     circuit.R('rp', 'rs_out', circuit.gnd, params['rp'])

     # define diode model
     circuit.model('d1', 'D', is_ = params['d1_is'], n = params['d1_n'], )

     # initialise single diode
     circuit.D('d1', 'rs_out', circuit.gnd, model = 'd1')


     # if 2-diode model
     if params['model'] == '2-diode':

         # define second diode model
         circuit.model('d2', 'D', is_ = params['d2_is'], n = params['d2_n'], )

         # initialise second diode
         circuit.D('d2', 'rs_out', circuit.gnd, model = 'd2')


     # if illuminated
     if params['light']:

         # initialise photocurrent source
         circuit.I('i1', circuit.gnd, 'rs_out', params['j'])

     # return circuit model
     return circuit

def low_bias_circuit(params):
     '''
         build equivalent circuit
     '''

     # initialise circuit
     circuit = Circuit('test_circuit')

     # initialise voltage source
     circuit.V('input', 'rs_out', circuit.gnd, 1@u_V)

     # initialise parallel resistance
     circuit.R('rp', 'rs_out', circuit.gnd, params['rp'])

     # if 2-diode model
     if params['model'] == '2-diode':

         # define second diode model
         circuit.model('d2', 'D', is_ = params['d2_is'], n = params['d2_n'], )

         # initialise second diode
         circuit.D('d2', 'rs_out', circuit.gnd, model = 'd2')


     # if illuminated
     if params['light']:

         # initialise photocurrent source
         circuit.I('i1', circuit.gnd, 'rs_out', params['j'])

     # return circuit model
     return circuit



def residuals_light(x, y, y_fit):
    '''
    x: independent variable
    y: dependent variable
    y_fit: fitting data
    '''
    #put fitting data together into a list
    data = [x,y,y_fit]

    #low bias error
    data_low = splice(data, maxval = 0.3)
    resid_low = 1000*(data_low[1]- data_low[2])**2

    #mpp error
    data_mpp = splice(data, minval = 0.3, maxval = 0.6)
    resid_mpp = 70*(data_mpp[1]- data_mpp[2])**2

    #high bias error
    data_high = splice(data, minval = 0.6)
    resid_high = (data_high[1]- data_high[2])**2


    residuals = np.append(resid_low,resid_mpp)
    residuals = np.append(residuals,resid_high)


    return residuals

def residuals_dark(x, y, y_fit):
    '''
    x: independent variable
    y: dependent variable
    y_fit: fitting data
    '''

    #put fitting data together into a list
    data = [ x , np.log10(abs(y)) , np.log10(abs(y_fit)) ]

    #low bias error
    data_low = splice(data, maxval = 0.3)
    resid_low = 100*(data_low[1]- data_low[2])**2

    #mpp error
    data_mpp = splice(data, minval = 0.3, maxval = 0.6)
    resid_mpp = 10*(data_mpp[1]- data_mpp[2])**2

    #high bias error
    data_high = splice(data, minval = 0.6)
    resid_high = 100*(data_high[1]- data_high[2])**2


    residuals = np.append(resid_low,resid_mpp)
    residuals = np.append(residuals,resid_high)


    return residuals

def low_bias_circuit_simulator(fit_params, params, V,I):

    # update model parameters for current state (optimisation variables)
    params['d2_is'] = np.power(10, fit_params[0])

    #params['d1_n'] = opt_vars[4]
    #params['d2_n'] = q/(k*(params['Temp']+273.15)*fit_params[2])
    params['rp'] = np.power(10, fit_params[1])

    if params['light']:
        params['j'] = fit_params[2]

    #generate equivalent circuit    
    model = low_bias_circuit(params)

    #simulate equivalent circuit with given parameters over voltage range
    I_fit = circuit_out(model, V, params['Temp'])

    #calculate residual of data fit
    if params['light']:
        residual = residuals_light(V,I, I_fit)
    
    else: 
        residual = residuals_dark(V,I,I_fit)
    

    return residual

def circuit_simulator(fit_params, params, V,I):

    # update model parameters for current state (optimisation variables)
    params['d1_is'] = np.power(10, fit_params[0])
    params['d2_is'] = np.power(10, fit_params[1])

    #params['d1_n'] = opt_vars[4]
    params['d2_n'] = q/(k*(params['Temp']+273.15)*fit_params[4])

    params['rs'] = fit_params[2]
    params['rp'] = np.power(10, fit_params[3])
    #params['rp'] = fit_params[3]
    #params['C'] = np.power(10,fit_params[4])

    if params['light']:
        params['j'] = fit_params[3]
    
    # else:
    #     params['C_t'] = np.power(10,fit_params[4])
    #generate equivalent circuit    
    model = circuit(params)

    #simulate equivalent circuit with given parameters over voltage range
    I_fit = circuit_out(model, V, params['Temp'])

    #calculate residual of data fit
    if params['light']:
        residual = residuals_light(V,I, I_fit)
    
    else: 
        residual = residuals_dark(V,I,I_fit)
    

    return residual



# define circuit parameters
params = {
     'model': '2-diode', # equivalent circuit model
     'Temp': 25, # temperature [C]

     'light': False, # illumination [bool]
     'tunnel': True, # tunnel diode used for D2
     'j': 0.04, # photocurrent [A/cm2]

     'd1_n': 1., # diode 1 ideality factor []
     'd1_is': 1e-15, # diode 1 saturation current [A/cm2]
     'd2_n': 2, # diode 2 ideality factor []
     'd2_is': 1e-5, # diode 2 saturation current [A/cm2]
     'rs': 0, # series resistance [Ohm-cm]
     'rp': 1e4, # shunt resistance [Ohm-cm]

     'C' : 1e-20 #capacitive constant
}

#random constants
k = 1.38e-23 #boltzmann constant in J/K
q = 1.602e-19 #elementary charge in coloumbs

'''FILE PATH AND NAME '''

#extract data from files
#filespath = r'C:\Users\Chris\OneDrive - UNSW\Thesis\Python\Data'
#filespath = r'C:\Users\Chris\OneDrive - UNSW\Thesis\Python\Data\SUNS VOC IV'
filespath = r'C:\Users\Chris\Documents\GitHub\shj-hons\data\loana-iv\dark\01\IV'
files = os.listdir(filespath)
#print(files)
#filesname = files[0]
filesname = r'01.drk'
print(filesname)



'''IV DATA EXTRACTION'''

###LOANA
data = LOANA_PARSE(filespath,filesname)
V = data.voltage
I = abs(data.current)/244.35

# ###SUNS VOC
# data = SUNS_VOC_PARSE(filespath, filesname)
# V = data.headers['V']
# I = abs(data.headers['I'])/244.35


###HALM IV
# data = iv.iv(file_type = 'halm-500', file_path = filespath, file_name = filesname)

# if params['light']:
#     _data = data['full']


# else:
#     _data = data['series']


# V = _data['voltage']
# I = -_data['current']/244.35


#sort data
V,I = array_sort(V,I)

P = V*I

# V_small, I_small = splice([V,I], minval = 0)
# dex = np.where(V_small == min(V_small))
# I_offset = I_small[dex]
# print(I_offset)
# params['C'] = float(abs(I_offset))

#slope, intercept, r_value, p_value, std_err = linregress(splice([V,np.log10(abs(I))], minval = 0.1, maxval = 0.3))

#print(slope)
#print(np.power(10,intercept))

#params['d2_n'] = slope

_V,_I = splice([V,I], minval = 0.1)

#SPLINE DATA AND ESTABLISH NEW X-RANGE #breaks when it comes to HJT files coz of duplicates in V, doesnt improve fit so removed
#spl = splrep(_V, _I)
# #V_new = np.linspace(V[0],0.3,200)
#I_new = splev(_V, spl)

# '''low bias fit'''

# #low bias fit guesses
# guess = [np.log10(params['d2_is']), np.log10(params['rp'])]
# if params['light']:
#     guess.append( params['j'] ) # photocurrent [A]

# #low bias fit bounds
# _bounds = [[np.log10( 1e-12 ),np.log10( 1e2 )],[np.log10( 1e-2 ), np.log10(1e5)]]
# if params['light']:
#      _bounds[0].append( 0 ) # min photocurrent [A]
#      _bounds[1].append( 0.05 ) # max photocurrent [A]


# #get low bias values

# V_,I_ = splice([V,I], minval = 0.1, maxval = 0.3)

# opt_low = least_squares(fun = low_bias_circuit_simulator, x0 = guess, bounds = _bounds, 
# args = (params, V_, I_), verbose = 2,
#                               ftol = 1e-12, xtol = 1e-12, gtol = 1e-15, 
# x_scale = 'jac')
# print(opt_low.x)


# params['d2_is'] = np.power(10, opt_low.x[0])
# params['rp'] = np.power(10, opt_low.x[1])

# if params['light']:
#     params['j'] = opt_low.x[2]



'''Optimisation Stage'''

# unpack initial values for optimisation variables
inits = [
     np.log10(params['d1_is']), # diode 1 saturation current [A]
     np.log10(params['d2_is']), # diode 1 saturation current [A]

     params['rs'], # series resistance [Ohm]
     np.log10(params['rp']), # shunt resistance [Ohm]
     #np.log10(params['C'])
     #params['rp'],
     #params['j'] # photocurrent [A]

     #params['d1_n'], # diode 1 ideality factor []
     q/(k*(params['Temp']+273.15)*params['d2_n']), # diode 2 ideality factor []
]
if params['light']:
     inits.append( params['j'] ) # photocurrent [A]


# define bounds on optimisation variables
bounds = [
     [
         np.log10( 1e-16 ), # min diode 1 saturation current [A]
         np.log10( 1e-12 ), # min diode 2 saturation current [A]
         0, # min series resistance [Ohm]
         np.log10( 1e2 ), # min shunt resistance [Ohm]
         #np.log10(1e-20)
         #1e2,
         #1., # diode 1 ideality factor []
         10, # diode 2 ideality factor []
     ],
     [
         np.log10( 1e-7 ), # max diode 1 saturation current [A]
         np.log10( 1e-2 ), # max diode 2 saturation current [A]
         2, # max series resistance [Ohm]
         np.log10(1e10), # max shunt resistance [Ohm]
         #np.log10(1e-2)
         #1e5,
         #1.3, # diode 1 ideality factor []
         100, # diode 2 ideality factor []
     ]
]
if params['light']:
     bounds[0].append( 0 ) # min photocurrent [A]
     bounds[1].append( 0.05 ) # max photocurrent [A]







# perform optimisation to fit equivalent circuit model to data
opt = least_squares(fun = circuit_simulator, x0 = inits, bounds = bounds, 
args = (params, _V, _I), verbose = 2,
                              ftol = 1e-12, xtol = 1e-12, gtol = 1e-12, 
x_scale = 'jac')

print(opt.x)

# generate equivalent circuit model
model = circuit(params)

_I = circuit_out(model, V, params['Temp'])

#fit quality
r_squared = R_squared(V, I,  _I)


#low bias fit

r_squared_low = R_squared(*splice([V,I,_I],0,0.3))

#high bias fit
r_squared_high = R_squared(*splice([V,np.log10(abs(I)),np.log10(abs(_I))],0.6))

#mpp fit
r_squared_mpp = R_squared(*splice([V,I,_I],0.3,0.6))

print(r_squared)
print(r_squared_low)
print(r_squared_mpp)
print(r_squared_high)

print('Fit parameters:\nJ01 = {0:5.3e} A,\nJ02 = {1:5.3e} A,\nRs = {2:5.3f} Ωcm2,\nRp = {3:5.3e} Ωcm2'.format(params['d1_is'], params['d2_is'], params['rs'], params['rp']))

#print('C = {0:5.3e} mA'.format(params['C']))

if params['light']:
        print('Jph = {0:5.3f} mA'.format(params['j']*1e3))
if params['tunnel']:
    print('A(T) =  {0:5.2f}'.format(q/(k*(params['Temp']+273.15)*params['d2_n'])))

# initialise figure
_w = 8; _h = 5; 
fig = plt.figure(figsize = (_w, _h))
fig.figsize  = [_w,_h]
ax = fig.add_subplot(111)


if params['light']:

     ax.scatter(V, I, s =10)

     ax.plot(V, _I, 'r-')

     ax.plot(V,P, 'g--')

     #ax.plot(V, P, 'y--')
     ax.set_ylim(-.01, 0.045)

else:
     ax.scatter(V, np.abs(I), s = 3)

     ax.plot(V, np.abs(_I), 'r-')

     ax.set_yscale('log')


ax.hlines(0., -.1, 1., linestyles = '--')
ax.vlines(0., -1., 0.045, linestyles = '--')

#ax.set_xlim(0, 0.8)



ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (A/cm2)')


plt.tight_layout()
plt.show()
