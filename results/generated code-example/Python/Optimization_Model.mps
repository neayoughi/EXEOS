* Signature: 0xac1ecdcb81a3ba43
NAME Optimization_Model
OBJSENSE MAX
ROWS
 N  OBJ
 L  Inventory[Comp-X1]
 L  Inventory[Comp-X2]
 L  Inventory[Comp-X3]
COLUMNS
    produce[FG-A]  OBJ       10
    produce[FG-A]  Inventory[Comp-X1]  1
    produce[FG-A]  Inventory[Comp-X2]  1
    produce[FG-B]  OBJ       15
    produce[FG-B]  Inventory[Comp-X1]  1
    produce[FG-B]  Inventory[Comp-X2]  2
    produce[FG-B]  Inventory[Comp-X3]  1
RHS
    RHS1      Inventory[Comp-X1]  8
    RHS1      Inventory[Comp-X2]  10
    RHS1      Inventory[Comp-X3]  3
BOUNDS
ENDATA
