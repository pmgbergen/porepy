h = 0.25*1e-2; //1e-1;
alpha = 1; //0.75*1e-1;

p0 = newp; Point(p0) = {0, 0, 0};
p1 = newp; Point(p1) = {1, 0, 0};
p2 = newp; Point(p2) = {1, 1, 0};
p3 = newp; Point(p3) = {0, 1, 0};

bound_line_0 = newl; Line(bound_line_0) = {p0, p1};
bound_line_1 = newl; Line(bound_line_1) = {p1, p2};
bound_line_2 = newl; Line(bound_line_2) = {p2, p3};
bound_line_3 = newl; Line(bound_line_3) = {p3, p0};

domain_loop = newll;
Line Loop(domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3};

domain_surf = news;
Plane Surface(domain_surf) = {domain_loop};

Physical Surface("DOMAIN") = {domain_surf};

p4 = newp; Point(p4) = {0.25, 0.5, 0};
p5 = newp; Point(p5) = {0.75, 0.5, 0};

inner_line = newl; Line(inner_line) = {p4, p5};
Physical Line("FRACTURE_0") = {inner_line};
Line{inner_line} In Surface{domain_surf};

Field[1] = Cylinder;
Field[1].Radius = 0.05;
Field[1].VIn = alpha*h;
Field[1].VOut = h;
Field[1].XCenter = 0.25;
Field[1].YCenter = 0.5;

Field[2] = Cylinder;
Field[2].Radius = 0.05;
Field[2].VIn = alpha*h;
Field[2].VOut = h;
Field[2].XCenter = 0.75;
Field[2].YCenter = 0.5;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;
