close all
clear all
clc

x = sym('x');
y = sym('y');
z = sym('z');

a = pi/4;

% Rotation about a unit vector
vect = [1, 0, 0];

vect = vect/norm(vect);
W = [[       0., -vect(3),  vect(2)]; ...
     [  vect(3),       0., -vect(1)]; ...
     [ -vect(2),  vect(1),       0. ]];
R = eye(3) + sin(a)*W + (1. - cos(a)) * W * W;

n = R*[0; 0; 1];
T = eye(3) - n*n';

p = symfun(x^2*z+4*y^2*sin(pi*y)-3*z^3, [x y z]);
K = symfun((1+(x^2+y^2))*eye(3), [x y z]);
%K = symfun((1+(x^2+y^2))*[1 0 0; 0 1 0; 0 0 0] + [0 0 0; 0 0 0; 0 0 1], [x y z]);
%K = symfun((1+100*(x^2+y^2+z^2))*[1 0 0; 0 1 0; 0 0 0] + [0 0 0; 0 0 0; 0 0 1], [x y z]);

grad_p = simplify(gradient(p(x, y, z), [x y z]));
u = simplify(-R'*K*R*T*grad_p);
rhs = simplify(divergence(u(x, y, z), [x y z]))
