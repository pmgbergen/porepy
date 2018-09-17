import numpy as np

# Domain size
x0 = 0
x1 = 1
y0 = 0
y1 = 1

# Fracture coordinates, runs from f_0 to f_1
f_0_x = x0
f_0_y = (y0 + y1) / 2
f_1_x = (x0 + x1) / 2
f_1_y = (y0 + y1) / 2

frac_hits_boundary = f_0_x == x0

# Size of the in- and outlet. Inlet will be located at
# x = x0, y\in [y0, inlet_outlet_size], while outlet is at
# x = x1, y\in [y - i1inlet_outlet_size, y1]
inlet_outlet_size = 0.1

# String formating
sx0 = str(x0)
sx1 = str(x1)
sy0 = str(y0)
sy1 = str(y1)


def write_geo(file_name, mesh_size=0.1):
    e = "};\n"
    h = str(mesh_size) + e

    s = "p00 = newp;"
    s += "Point(p00) = {" + sx0 + ", " + sy0 + ", 0, " + h

    s += "p10 = newp;"
    s += "Point(p10) = {" + sx1 + ", " + sy0 + ", 0, " + h

    s += "p11 = newp;"
    s += "Point(p11) = {" + sx1 + ", " + sy1 + ", 0, " + h

    s += "p01 = newp;"
    s += "Point(p01) = {" + sx0 + ", " + sy1 + ", 0, " + h

    s += "p0i = newp;"
    s += "Point(p0i) = {" + sx0 + ", " + str(y0 + inlet_outlet_size) + ", 0, " + h

    s += "p1i = newp;"
    s += "Point(p1i) = {" + sx1 + ", " + str(y1 - inlet_outlet_size) + ", 0, " + h

    s += "pf0 = newp;"
    s += "Point(pf0) = {" + str(f_0_x) + ", " + str(f_0_y) + ", 0, " + h

    s += "pf1 = newp;"
    s += "Point(pf1) = {" + str(f_1_x) + ", " + str(f_1_y) + ", 0, " + h

    s += "bottom = newl;"
    s += "Line(bottom) = {p00, p10};\n"

    s += "right = newl;"
    s += "Line(right) = {p10, p1i};\n"

    s += "outlet = newl;"
    s += "Line(outlet) = {p1i, p11};\n"

    s += "top = newl;"
    s += "Line(top) = {p11, p01};\n"

    s += "left_up = newl;"
    s += "Line(left_up) = {p01, pf0};\n"

    s += "left_d = newl;"
    s += "Line(left_d) = {pf0, p0i};\n"

    s += "inlet = newl;"
    s += "Line(inlet) = {p0i, p00};\n"

    s += "bound = newll;\n"
    s += "Line Loop(bound) = {bottom, right, outlet, top, left_up, left_d,"
    s += "inlet};\n\n"

    s += "domain = news;\n"
    s += "Plane Surface(domain) = {bound};\n"
    s += 'Physical Surface("DOMAIN") = {domain};\n'

    s += "frac = newl;"
    s += "Line(frac) = {pf0, pf1};\n"
    s += "Line{frac} In Surface{domain};"
    s += 'Physical Line("FRACTURE_1") = {frac};\n'

    with open(file_name, "w") as f:
        f.write(s)


if __name__ == "__main__":
    write_geo("hei.geo")
