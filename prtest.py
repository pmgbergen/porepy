import porepy as pp


PR = pp.composite.PR_Composition()

print("---")
print("discriminant: ", PR.discriminant)
print("D: ", PR._D)
print("---")
