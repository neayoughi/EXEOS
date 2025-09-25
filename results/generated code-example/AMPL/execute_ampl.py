#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from amplpy import AMPL
ampl = AMPL()
ampl.reset()
ampl.set_option('reset_initial_guesses', True)
ampl.set_option('send_statuses', False)
ampl.read('model.mod')
ampl.read_data('data.dat')
ampl.set_option('solver', 'gurobi')
ampl.solve()
variables = ampl.getVariables()
result_str = ""
for var_tuple in variables:
    var_name = var_tuple[0]
    var = variables[var_name]
    result_str += f"Variable: {var_name}\n"
    values = var.getValues()
    result_str += values.toString() + "\n\n"
print(result_str)
