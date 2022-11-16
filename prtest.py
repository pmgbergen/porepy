import porepy as pp


PR = pp.composite.PengRobinsonComposition()

print("---")
print("Z1: ", PR.Z1)
print("Z2: ", PR.Z2)
print("Z3: ", PR.Z3)
print("---")
print("discriminant: ", PR.discriminant)
print("D: ", PR._D)
print("---")
print("S-T: ", PR._S - PR._T)