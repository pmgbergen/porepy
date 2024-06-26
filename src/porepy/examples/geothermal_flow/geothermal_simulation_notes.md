# Geothermal simulation with Dreisner correlations

The following notes contain essential information on the implementation of a geothermal simulation with brine correlations.

## Critial components for no-gravity simulations

- MD consistent pressure equation:
	- Fixed dimensional case
	- Mixed-dimensional case	
- Write down the formulation with the notation in the overleaf document
- Setting up simple case with:
	-  IC along x: Linear pressure and enthlapy, constant initialization of z_NaCL at 0.001
	-  Dirichlet BC data
	-  Inlet BC z_NaCL = 0.01
	-  Domain aspect ratio = dx/dy = 10

## Characterization of a linear tracer setting (Isothermal)
	- M_H2O = 0.01801528 kg/mol
	- M_NaCL = 0.01801528 kg/mol (Linear tracer) = 58.44 g/mol
	- x_NaCL_L = z_NaCL
	- x_H20_L = z_H2O
	- x_NaCL_V = z_NaCL
	- x_H20_V = z_H2O
	- rho_L = 55508.435061792 mol/m3
	- rho_V = 55.5084350618 mol/m3
	- hat{rho}_L = 1000 kg / m3
	- hat{rho}_V = 1 kg / m3
	- mu_L = 0.001 Pa s
	- mu_V = 0.00001 Pa s 
	- hat{rho}_H2O = hat{rho}_L
	- hat{rho}_NaCL = hat{rho}_L
	- S_L = 1.0
	- S_V = 1.0 - S_L = 0.0
	- rho = S_L rho_L + S_V rho_V
	- h = h_L as variable
	- h_V = 0.0
	- T approx h_L and constant
	- K_e = Identity
	- h_s = 0.0
	- v_H2O = 1 / rho_L
	- v_NaCL = 1 / rho_L 
