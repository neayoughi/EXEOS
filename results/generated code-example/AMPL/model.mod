# SETS
set PROD; # Set of finished products
set COMP; # Set of components
# PARAMETERS
param price {j in PROD} >= 0; # Selling price per unit of each product
param inventory {i in COMP} >= 0; # On-hand inventory for each component
param bom {i in COMP, j in PROD} >= 0; # Bill of Materials: units of component i per unit of product j
# VARIABLES
var Produce {j in PROD} >= 0; # Number of units of each product to manufacture
# OBJECTIVE
maximize Total_Revenue:
sum {j in PROD} (price[j] * Produce[j]);
# CONSTRAINTS
# Constraint to ensure component usage does not exceed available inventory
subject to Inventory_Limit {i in COMP}:
(sum {j in PROD} (bom[i,j] * Produce[j])) <= inventory[i]