///////////////////////////////////////////////////////////////////
// Gmsh geometry file for the domain used in Barlag 1998
// The 3d domain consists of two layers and a 2d fault zone
// embedded in the upper one. Meshing is done using tetrahedra.
///////////////////////////////////////////////////////////////////

ref = 1.0;                    // refinement in z-direction towards fracture plane (ref < 1.0)
numPointsZ_layer1 = 2;       // no. vertices in z in layer one
numPointsZ_layer2_above = 5; // no. vertices in z in layer two above the fault
numPointsZ_layer2_below = 5; // no. vertices in z in layer two below the fault
numPointsX = 10;              // no. vertices in x direction
numPointsY = 10;              // no. vertices in y direction

// domain bounding box
Point(1) = {0.0, 0.0, 0.0, 1.0};
Point(2) = {100.0, 0.0, 0.0, 1.0};
Point(3) = {100.0, 100.0, 0.0, 1.0};
Point(4) = {0.0, 100.0, 0.0, 1.0};
Point(5) = {0.0, 0.0, 100.0, 1.0};
Point(6) = {100.0, 0.0, 100.0, 1.0};
Point(7) = {100.0, 100.0, 100.0, 1.0};
Point(8) = {0.0, 100.0, 100.0, 1.0};

// Lower layer boundary points
Point(9) = {0.0, 0.0, 10.0, 1.0};
Point(10) = {100.0, 0.0, 10.0, 1.0};
Point(11) = {100.0, 100.0, 10.0, 1.0};
Point(12) = {0.0, 100.0, 10.0, 1.0};

// fault zone boundary points
Point(13) = {0.0, 0.0, 80.0, 1.0};
Point(14) = {100.0, 0.0, 20.0, 1.0};
Point(15) = {100.0, 100.0, 20.0, 1.0};
Point(16) = {0.0, 100.0, 80.0, 1.0};

// layer for boundary condition on vertical discretization
Point(17) = {0.0, 0.0, 90.0, 1.0};
Point(18) = {0.0, 100.0, 90.0, 1.0};

// layer one vertical discretization
Line(1) = {1, 9};
Line(2) = {4, 12};
Line(3) = {2, 10};
Line(4) = {3, 11};
Transfinite Line{1:4} = numPointsZ_layer1;

// layer two vertical discretization below the fault
Line(5) = {10, 14};
Line(6) = {11, 15};
Line(7) = {12, 16};
Line(8) = {9, 13};
Transfinite Line{5:8} = numPointsZ_layer2_below Using Progression ref;

// layer two vertical discretization above the fault
Line(9) = {5, 17};
Line(109) = {17, 13};
Line(10) = {8, 18};
Line(110) = {18, 16};
Line(11) = {6, 14};
Line(12) = {7, 15};
Transfinite Line{9,109,10,110,11,12} = numPointsZ_layer2_above Using Progression ref;

// discretization in x-direction
Line(13) = {1, 2};
Line(14) = {9, 10};
Line(15) = {13, 14};
Line(16) = {5, 6};
Line(17) = {4, 3};
Line(18) = {12, 11};
Line(19) = {16, 15};
Line(20) = {8, 7};
Transfinite Line{13:20} = numPointsX;

// discretization in y-direction
Line(21) = {2, 3};
Line(22) = {10, 11};
Line(23) = {14, 15};
Line(24) = {6, 7};
Line(25) = {1, 4};
Line(26) = {9, 12};
Line(27) = {13, 16};
Line(28) = {5, 8};
Line(112) = {17, 18};
Transfinite Line{21:28,112} = numPointsY;

// lower layer volume
Line Loop(29) = {1, 14, -3, -13};
Plane Surface(30) = {29};
Line Loop(31) = {3, 22, -4, -21};
Plane Surface(32) = {31};
Line Loop(33) = {4, -18, -2, 17};
Plane Surface(34) = {33};
Line Loop(35) = {2, -26, -1, 25};
Plane Surface(36) = {35};
Line Loop(37) = {25, 17, -21, -13};
Plane Surface(38) = {37};
Line Loop(39) = {26, 18, -22, -14};
Plane Surface(40) = {39};
Surface Loop(41) = {40, 36, 34, 32, 30, 38};
Volume(42) = {41};

// upper layer volumes
Line Loop(43) = {8, 15, -5, -14};
Plane Surface(44) = {43};
Line Loop(45) = {5, 23, -6, -22};
Plane Surface(46) = {45};
Line Loop(47) = {18, 6, -19, -7};
Plane Surface(48) = {47};
Line Loop(49) = {7, -27, -8, 26};
Plane Surface(50) = {49};
Line Loop(51) = {15, 23, -19, -27};
Plane Surface(52) = {51}; // fault plane
Surface Loop(53) = {52, 44, 50, 48, 46, 40};
Volume(54) = {53};
Line Loop(55) = {15, -11, -16, 9, 109};
Plane Surface(56) = {55};
Line Loop(57) = {11, 23, -12, -24};
Plane Surface(58) = {57};
Line Loop(59) = {12, -19, -10, -110, 20};
Plane Surface(60) = {59};
Line Loop(61) = {10, -112, -9, 28};
Plane Surface(62) = {61};
Line Loop(161) = {110, -27, -109, 112};
Plane Surface(162) = {161};
Line Loop(63) = {28, 20, -24, -16};
Plane Surface(64) = {63};
Surface Loop(65) = {64, 62, 162, 60, 58, 56, 52};
Volume(66) = {65};

// use hexaheders for meshing
//Transfinite Surface "*";
//Recombine Surface "*";
//Transfinite Volume "*";

// give physical entity indices to layers
Physical Volume(1) = {42};     // lower layer
Physical Volume(2) = {54, 66}; // upper layer
Physical Surface("FRACTURE_0") = {52};    // fault zone
