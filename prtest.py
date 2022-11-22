import porepy as pp


PR = pp.composite.PR_Composition()

H2O = pp.composite.H2O(PR.ad_system)
print(H2O.name)
print("---")
print("discriminant: ", PR.discriminant)
print("D: ", PR._D)
print("---")
