import gurobipy as gp
from gurobipy import GRB
import json
with open("input.json", "r") as file:
    data = json.load(file)
products = [p['Product'] for p in data['Products']]
components = [c['Component'] for c in data['Components']]
prices = {p['Product']: p['SellingPrice'] for p in data['Prices']}
inventory = {i['Component']: i['OnHand'] for i in data['Inventory']}
bom = {p: {c: 0 for c in components} for p in products}
for item in data['BOM']:
    bom[item['Product']][item['Component']] = item['UnitsRequired']
model = gp.Model("Optimization_Model")
produce = model.addVars(products, vtype=GRB.CONTINUOUS, name="produce")
model.setObjective(
    gp.quicksum(prices[p] * produce[p] for p in products),
    GRB.MAXIMIZE
)
model.addConstrs(
    (gp.quicksum(bom[p][c] * produce[p] for p in products) <= inventory[c]
     for c in components),
    name="Inventory"
)
model.optimize()
if model.status == GRB.OPTIMAL:
    print("Optimal Objective Value:", model.objVal)
    for v in model.getVars():
        if v.X > 1e-6:
            print(v.varName, v.X)
else:
    print("No feasible solution found.")
model.write("Optimization_Model.mps")