close all
clear all
clc

x = sym('x');
y = sym('y');
z = sym('z');

a = sym('a');
a = pi/4.;
R = [1 0 0;0 cos(a) -sin(a); 0 sin(a) cos(a)];
n = R*[0; 0; 1];
T = eye(3) - n*n';

p = symfun(x^2*z+4*y^2*sin(pi*y)-3*z^3, [x y z]);
K = symfun(1+x^2+y^2, [x y z]);

grad_p = gradient(p(x, y, z), [x y z]);
u = simplify(-K*T*grad_p);
rhs = simplify(divergence(u(x, y, z), [x y z]))
