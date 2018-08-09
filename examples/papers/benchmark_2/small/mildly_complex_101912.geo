Geometry.Tolerance = 3.33333333333e-05;

// Mesh size parameters. Roughly grouped into classes with aparently similar geometric complexity.
// Feel free to refine this.

// Domain corners
h_domain = 0.07;

// Fracture 1, left (low x) and right ends
h_1_left = 0.05;
h_1_right = 0.05;

// Fracture 2
h_2 = 0.05;

// Intersection of 1 and 2
h_1_2 = 0.05;

// Intersection of fractures 1 and 3
h_1_3 = 0.025;
// Other points on fracture 3
h_3 = 0.05;

// Points on fracture 4, close to f1 and far away
h_4_close = 0.025;
h_4_away = 0.05;

// Endpoints of fractures 5 and 6. Not intersection (below)
h_5_6 = 0.04;
// Intersection between 5 and 6
h_5_6_isect = 0.025;

// Intersection of 1 with 5 and 6
h_1_5_6 = 0.025;

// Fracture 7
h_7 = 0.04;
// Fracture 8
h_8 = 0.04;
// Intersection between 1 and 7 and 8
h_1_7 = 0.03;
h_1_8 = 0.03;

// Define points
ymax = 2.25;

// Fracture 1
p0 = newp; Point(p0) = {0.05, 0.25, 0.5, h_1_left };
p1 = newp; Point(p1) = {0.95, 0.25, 0.5, h_1_left };
p2 = newp; Point(p2) = {0.95, 2.0, 0.5, h_1_right };
p3 = newp; Point(p3) = {0.05, 2.0, 0.5, h_1_right };
// Fracture 2
p4 = newp; Point(p4) = {0.5, 0.05, 0.95, h_2 };
p5 = newp; Point(p5) = {0.5, 0.05, 0.05, h_2 };
p6 = newp; Point(p6) = {0.5, 0.3, 0.05, h_2 };
p7 = newp; Point(p7) = {0.5, 0.3, 0.95, h_2 };
// Intersection of fracture 1 and fracture 3
p8 = newp; Point(p8) = {0.05, 1.0, 0.5, h_1_3 };
p9 = newp; Point(p9) = {0.95, 1.0, 0.5, h_1_3 };
// Other points of fracture 3
p10 = newp; Point(p10) = {0.95, 2.2, 0.85, h_3 };
p11 = newp; Point(p11) = {0.05, 2.2, 0.85, h_3 };
// Fracture 4
p12 = newp; Point(p12) = {0.05, 1.0, 0.48, h_4_close };
p13 = newp; Point(p13) = {0.95, 1.0, 0.48, h_4_close };
p14 = newp; Point(p14) = {0.95, 2.2, 0.14, h_4_away };
p15 = newp; Point(p15) = {0.05, 2.2, 0.14, h_4_away };
// Fractures 5 and 6
p16 = newp; Point(p16) = {0.23, 1.9, 0.3, h_5_6};
p17 = newp; Point(p17) = {0.23, 1.9, 0.7, h_5_6};
p18 = newp; Point(p18) = {0.17, 2.2, 0.7, h_5_6};
p19 = newp; Point(p19) = {0.17, 2.2, 0.3, h_5_6};
p20 = newp; Point(p20) = {0.17, 1.9, 0.3, h_5_6};
p21 = newp; Point(p21) = {0.17, 1.9, 0.7, h_5_6};
p22 = newp; Point(p22) = {0.23, 2.2, 0.7, h_5_6};
p23 = newp; Point(p23) = {0.23, 2.2, 0.3, h_5_6};
// Fracture 7
p24 = newp; Point(p24) = {.77, 1.9, 0.3, h_7 };
p25 = newp; Point(p25) = {.77, 1.9, 0.7, h_7 };
p26 = newp; Point(p26) = {.77, 2.2, 0.7, h_7 };
p27 = newp; Point(p27) = {.77, 2.2, 0.3, h_7 };
// Fracture 8
p28 = newp; Point(p28) = {0.83, 1.9, 0.3, h_8 };
p29 = newp; Point(p29) = {0.83, 1.9, 0.7, h_8 };
p30 = newp; Point(p30) = {0.83, 2.2, 0.7, h_8 };
p31 = newp; Point(p31) = {0.83, 2.2, 0.3, h_8 };
// Domain corners
p32 = newp; Point(p32) = {0.0, 0.0, 1.0, h_domain };
p33 = newp; Point(p33) = {0.0, 0.0, 0.0, h_domain };
p34 = newp; Point(p34) = {0.0, ymax, 0.0, h_domain };
p35 = newp; Point(p35) = {0.0, ymax, 1.0, h_domain };
p36 = newp; Point(p36) = {1.0, 0.0, 1.0, h_domain };
p37 = newp; Point(p37) = {1.0, 0.0, 0.0, h_domain };
p38 = newp; Point(p38) = {1.0, ymax, 0.0, h_domain };
p39 = newp; Point(p39) = {1.0, ymax, 1.0, h_domain };
// Intersection of Fracture 1 and 2
p40 = newp; Point(p40) = {0.5, 0.3, 0.5, h_1_2 };
p41 = newp; Point(p41) = {0.5, 0.25, 0.5, h_1_2 };
// Intersections between 1 and 5 and 6
p42 = newp; Point(p42) = {0.23, 1.9, 0.5, h_1_5_6 };
p43 = newp; Point(p43) = {0.21, 2.0, 0.5, h_1_5_6 };
p44 = newp; Point(p44) = {0.17, 1.9, 0.5, h_1_5_6 };
p45 = newp; Point(p45) = {0.19, 2.0, 0.5, h_1_5_6 };
// Intersections between 1 and 7 and 8, respectively
p46 = newp; Point(p46) = {0.77, 1.9, 0.5, h_1_7 };
p47 = newp; Point(p47) = {0.77, 2.0, 0.5, h_1_7 };
p48 = newp; Point(p48) = {0.83, 1.9, 0.5, h_1_8 };
p49 = newp; Point(p49) = {0.83, 2.0, 0.5, h_1_8 };
// Intersection of 5 and 6
p50 = newp; Point(p50) = {0.2, 2.05, 0.7, h_5_6_isect };
p51 = newp; Point(p51) = {0.2, 2.05, 0.3, h_5_6_isect };

// Points added to the domain boundaries to allow specification of boundary conditions
pin00 = newp; Point(pin00) = {0., 0., 0.333333, h_domain};
pin01 = newp; Point(pin01) = {0., 0., 0.666667, h_domain};
pin11 = newp; Point(pin11) = {1., 0., 0.666667, h_domain};
pin10 = newp; Point(pin10) = {1., 0., 0.333333, h_domain};
pout00 = newp; Point(pout00) = {0., ymax, 0.333333, h_domain};
pout01 = newp; Point(pout01) = {0., ymax, 0.666667, h_domain};
pout11 = newp; Point(pout11) = {1., ymax, 0.666667, h_domain};
pout10 = newp; Point(pout10) = {1., ymax, 0.333333, h_domain};

// End of point specification

// Define lines 
frac_line_0= newl; Line(frac_line_0) = {p0, p8};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1= newl; Line(frac_line_1) = {p0, p41};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2= newl; Line(frac_line_2) = {p1, p9};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3= newl; Line(frac_line_3) = {p1, p41};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4= newl; Line(frac_line_4) = {p2, p9};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5= newl; Line(frac_line_5) = {p2, p49};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6= newl; Line(frac_line_6) = {p3, p8};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7= newl; Line(frac_line_7) = {p3, p45};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8= newl; Line(frac_line_8) = {p4, p5};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9= newl; Line(frac_line_9) = {p4, p7};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10= newl; Line(frac_line_10) = {p5, p6};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11= newl; Line(frac_line_11) = {p6, p40};
Physical Line("FRACTURE_TIP_11") = {frac_line_11};

frac_line_12= newl; Line(frac_line_12) = {p7, p40};
Physical Line("FRACTURE_TIP_12") = {frac_line_12};

frac_line_13= newl; Line(frac_line_13) = {p8, p9};
Physical Line("FRACTURE_LINE_13") = {frac_line_13};

frac_line_14= newl; Line(frac_line_14) = {p8, p11};
Physical Line("FRACTURE_TIP_14") = {frac_line_14};

frac_line_15= newl; Line(frac_line_15) = {p9, p10};
Physical Line("FRACTURE_TIP_15") = {frac_line_15};

frac_line_16= newl; Line(frac_line_16) = {p10, p11};
Physical Line("FRACTURE_TIP_16") = {frac_line_16};

frac_line_17= newl; Line(frac_line_17) = {p12, p13};
Physical Line("FRACTURE_TIP_17") = {frac_line_17};

frac_line_18= newl; Line(frac_line_18) = {p12, p15};
Physical Line("FRACTURE_TIP_18") = {frac_line_18};

frac_line_19= newl; Line(frac_line_19) = {p13, p14};
Physical Line("FRACTURE_TIP_19") = {frac_line_19};

frac_line_20= newl; Line(frac_line_20) = {p14, p15};
Physical Line("FRACTURE_TIP_20") = {frac_line_20};

frac_line_21= newl; Line(frac_line_21) = {p16, p42};
Physical Line("FRACTURE_TIP_21") = {frac_line_21};

frac_line_22= newl; Line(frac_line_22) = {p16, p51};
Physical Line("FRACTURE_TIP_22") = {frac_line_22};

frac_line_23= newl; Line(frac_line_23) = {p17, p42};
Physical Line("FRACTURE_TIP_23") = {frac_line_23};

frac_line_24= newl; Line(frac_line_24) = {p17, p50};
Physical Line("FRACTURE_TIP_24") = {frac_line_24};

frac_line_25= newl; Line(frac_line_25) = {p18, p19};
Physical Line("FRACTURE_TIP_25") = {frac_line_25};

frac_line_26= newl; Line(frac_line_26) = {p18, p50};
Physical Line("FRACTURE_TIP_26") = {frac_line_26};

frac_line_27= newl; Line(frac_line_27) = {p19, p51};
Physical Line("FRACTURE_TIP_27") = {frac_line_27};

frac_line_28= newl; Line(frac_line_28) = {p20, p44};
Physical Line("FRACTURE_TIP_28") = {frac_line_28};

frac_line_29= newl; Line(frac_line_29) = {p20, p51};
Physical Line("FRACTURE_TIP_29") = {frac_line_29};

frac_line_30= newl; Line(frac_line_30) = {p21, p44};
Physical Line("FRACTURE_TIP_30") = {frac_line_30};

frac_line_31= newl; Line(frac_line_31) = {p21, p50};
Physical Line("FRACTURE_TIP_31") = {frac_line_31};

frac_line_32= newl; Line(frac_line_32) = {p22, p23};
Physical Line("FRACTURE_TIP_32") = {frac_line_32};

frac_line_33= newl; Line(frac_line_33) = {p22, p50};
Physical Line("FRACTURE_TIP_33") = {frac_line_33};

frac_line_34= newl; Line(frac_line_34) = {p23, p51};
Physical Line("FRACTURE_TIP_34") = {frac_line_34};

frac_line_35= newl; Line(frac_line_35) = {p24, p27};
Physical Line("FRACTURE_TIP_35") = {frac_line_35};

frac_line_36= newl; Line(frac_line_36) = {p24, p46};
Physical Line("FRACTURE_TIP_36") = {frac_line_36};

frac_line_37= newl; Line(frac_line_37) = {p25, p26};
Physical Line("FRACTURE_TIP_37") = {frac_line_37};

frac_line_38= newl; Line(frac_line_38) = {p25, p46};
Physical Line("FRACTURE_TIP_38") = {frac_line_38};

frac_line_39= newl; Line(frac_line_39) = {p26, p27};
Physical Line("FRACTURE_TIP_39") = {frac_line_39};

frac_line_40= newl; Line(frac_line_40) = {p28, p31};
Physical Line("FRACTURE_TIP_40") = {frac_line_40};

frac_line_41= newl; Line(frac_line_41) = {p28, p48};
Physical Line("FRACTURE_TIP_41") = {frac_line_41};

frac_line_42= newl; Line(frac_line_42) = {p29, p30};
Physical Line("FRACTURE_TIP_42") = {frac_line_42};

frac_line_43= newl; Line(frac_line_43) = {p29, p48};
Physical Line("FRACTURE_TIP_43") = {frac_line_43};

frac_line_44= newl; Line(frac_line_44) = {p30, p31};
Physical Line("FRACTURE_TIP_44") = {frac_line_44};

frac_line_45= newl; Line(frac_line_45) = {p32, pin01};
Physical Line("AUXILIARY_LINE_45") = {frac_line_45};
frac_line_45_1= newl; Line(frac_line_45_1) = {pin01, pin00};
Physical Line("AUXILIARY_LINE_45_1") = {frac_line_45_1};
frac_line_45_2= newl; Line(frac_line_45_2) = {pin00, p33};
Physical Line("AUXILIARY_LINE_45_2") = {frac_line_45_2};

frac_line_46= newl; Line(frac_line_46) = {p32, p35};
Physical Line("AUXILIARY_LINE_46") = {frac_line_46};

frac_line_47= newl; Line(frac_line_47) = {p32, p36};
Physical Line("AUXILIARY_LINE_47") = {frac_line_47};

frac_line_48= newl; Line(frac_line_48) = {p33, p34};
Physical Line("AUXILIARY_LINE_48") = {frac_line_48};

frac_line_49= newl; Line(frac_line_49) = {p33, p37};
Physical Line("AUXILIARY_LINE_49") = {frac_line_49};

frac_line_50= newl; Line(frac_line_50) = {p34, pout00};
Physical Line("AUXILIARY_LINE_50") = {frac_line_50};
frac_line_50_1= newl; Line(frac_line_50_1) = {pout00, pout01};
Physical Line("AUXILIARY_LINE_50_1") = {frac_line_50_1};
frac_line_50_2= newl; Line(frac_line_50_2) = {pout01, p35};
Physical Line("AUXILIARY_LINE_50_2") = {frac_line_50_1};

frac_line_51= newl; Line(frac_line_51) = {p34, p38};
Physical Line("AUXILIARY_LINE_51") = {frac_line_51};

frac_line_52= newl; Line(frac_line_52) = {p35, p39};
Physical Line("AUXILIARY_LINE_52") = {frac_line_52};

frac_line_53= newl; Line(frac_line_53) = {p36, pin11};
Physical Line("AUXILIARY_LINE_53") = {frac_line_53};
frac_line_53_1= newl; Line(frac_line_53_1) = {pin11, pin10};
Physical Line("AUXILIARY_LINE_53_1") = {frac_line_53_1};
frac_line_53_2= newl; Line(frac_line_53_2) = {pin10, p37};
Physical Line("AUXILIARY_LINE_53_2") = {frac_line_53_2};

frac_line_54= newl; Line(frac_line_54) = {p36, p39};
Physical Line("AUXILIARY_LINE_54") = {frac_line_54};

frac_line_55= newl; Line(frac_line_55) = {p37, p38};
Physical Line("AUXILIARY_LINE_55") = {frac_line_55};

frac_line_56= newl; Line(frac_line_56) = {p38, pout10};
Physical Line("AUXILIARY_LINE_56") = {frac_line_56};
frac_line_56_1= newl; Line(frac_line_56_1) = {pout10, pout11};
Physical Line("AUXILIARY_LINE_56_1") = {frac_line_56_1};
frac_line_56_2= newl; Line(frac_line_56_2) = {pout11, p39};
Physical Line("AUXILIARY_LINE_56_2") = {frac_line_56_2};

frac_line_57= newl; Line(frac_line_57) = {p40, p41};
Physical Line("FRACTURE_LINE_57") = {frac_line_57};

frac_line_58= newl; Line(frac_line_58) = {p42, p43};
Physical Line("FRACTURE_LINE_58") = {frac_line_58};

frac_line_59= newl; Line(frac_line_59) = {p43, p45};
Physical Line("FRACTURE_TIP_59") = {frac_line_59};

frac_line_60= newl; Line(frac_line_60) = {p43, p47};
Physical Line("FRACTURE_TIP_60") = {frac_line_60};

frac_line_61= newl; Line(frac_line_61) = {p44, p45};
Physical Line("FRACTURE_LINE_61") = {frac_line_61};

frac_line_62= newl; Line(frac_line_62) = {p46, p47};
Physical Line("FRACTURE_LINE_62") = {frac_line_62};

frac_line_63= newl; Line(frac_line_63) = {p47, p49};
Physical Line("FRACTURE_TIP_63") = {frac_line_63};

frac_line_64= newl; Line(frac_line_64) = {p48, p49};
Physical Line("FRACTURE_LINE_64") = {frac_line_64};

frac_line_65= newl; Line(frac_line_65) = {p50, p51};
Physical Line("FRACTURE_LINE_65") = {frac_line_65};


in_line_low = newl; Line(in_line_low) = {pin00, pin10};
Physical Line("InLineLow") = {in_line_low};
in_line_hi = newl; Line(in_line_hi) = {pin01, pin11};
Physical Line("InLineHigh") = {in_line_hi};
out_line_low = newl; Line(out_line_low) = {pout00, pout10};
Physical Line("OutLineLow") = {out_line_low};
out_line_hi = newl; Line(out_line_hi) = {pout01, pout11};
Physical Line("OutLineHigh") = {out_line_hi};

// End of line specification 

// Start domain specification
frac_loop_8 = newll; 
Line Loop(frac_loop_8) = { frac_line_45, frac_line_45_1, frac_line_45_2, frac_line_48, frac_line_50, frac_line_50_1, frac_line_50_2, -frac_line_46};
auxiliary_8 = news; Plane Surface(auxiliary_8) = {frac_loop_8};
Physical Surface("AUXILIARY_8") = {auxiliary_8};

frac_loop_9 = newll; 
Line Loop(frac_loop_9) = { frac_line_53, frac_line_53_1, frac_line_53_2, frac_line_55, frac_line_56, frac_line_56_1, frac_line_56_2, -frac_line_54};
auxiliary_9 = news; Plane Surface(auxiliary_9) = {frac_loop_9};
Physical Surface("AUXILIARY_9") = {auxiliary_9};

frac_loop_10 = newll; 
Line Loop(frac_loop_10) = { frac_line_45,frac_line_45_1 ,frac_line_45_2, frac_line_49, -frac_line_53, -frac_line_53_1, -frac_line_53_2, -frac_line_47};
auxiliary_10 = news; Plane Surface(auxiliary_10) = {frac_loop_10};
Physical Surface("AUXILIARY_10") = {auxiliary_10};

frac_loop_11 = newll; 
Line Loop(frac_loop_11) = { frac_line_50, frac_line_50_1, frac_line_50_2, frac_line_52, -frac_line_56, -frac_line_56_1, -frac_line_56_2, -frac_line_51};
auxiliary_11 = news; Plane Surface(auxiliary_11) = {frac_loop_11};
Physical Surface("AUXILIARY_11") = {auxiliary_11};

frac_loop_12 = newll; 
Line Loop(frac_loop_12) = { frac_line_48, frac_line_51, -frac_line_55, -frac_line_49};
auxiliary_12 = news; Plane Surface(auxiliary_12) = {frac_loop_12};
Physical Surface("AUXILIARY_12") = {auxiliary_12};

frac_loop_13 = newll; 
Line Loop(frac_loop_13) = { frac_line_46, frac_line_52, -frac_line_54, -frac_line_47};
auxiliary_13 = news; Plane Surface(auxiliary_13) = {frac_loop_13};
Physical Surface("AUXILIARY_13") = {auxiliary_13};

domain_loop = newsl;
Surface Loop(domain_loop) = {auxiliary_8,auxiliary_9,auxiliary_10,auxiliary_11,auxiliary_12,auxiliary_13};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, -frac_line_6, frac_line_7, -frac_line_59, frac_line_60, frac_line_63, -frac_line_5, frac_line_4, -frac_line_2, frac_line_3, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};

Line{frac_line_13} In Surface{fracture_0};
Line{frac_line_57} In Surface{fracture_0};
Line{frac_line_58} In Surface{fracture_0};
Line{frac_line_61} In Surface{fracture_0};
Line{frac_line_62} In Surface{fracture_0};
Line{frac_line_64} In Surface{fracture_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_8, frac_line_10, frac_line_11, -frac_line_12, -frac_line_9};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};

Line{frac_line_57} In Surface{fracture_1};

frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_13, frac_line_15, frac_line_16, -frac_line_14};
fracture_2 = news; Plane Surface(fracture_2) = {frac_loop_2};
Physical Surface("FRACTURE_2") = {fracture_2};
Surface{fracture_2} In Volume{1};


frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_17, frac_line_19, frac_line_20, -frac_line_18};
fracture_3 = news; Plane Surface(fracture_3) = {frac_loop_3};
Physical Surface("FRACTURE_3") = {fracture_3};
Surface{fracture_3} In Volume{1};


frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_21, -frac_line_23, frac_line_24, -frac_line_26, frac_line_25, frac_line_27, -frac_line_22};
fracture_4 = news; Plane Surface(fracture_4) = {frac_loop_4};
Physical Surface("FRACTURE_4") = {fracture_4};
Surface{fracture_4} In Volume{1};

Line{frac_line_58} In Surface{fracture_4};
Line{frac_line_65} In Surface{fracture_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_28, -frac_line_30, frac_line_31, -frac_line_33, frac_line_32, frac_line_34, -frac_line_29};
fracture_5 = news; Plane Surface(fracture_5) = {frac_loop_5};
Physical Surface("FRACTURE_5") = {fracture_5};
Surface{fracture_5} In Volume{1};

Line{frac_line_61} In Surface{fracture_5};
Line{frac_line_65} In Surface{fracture_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_35, -frac_line_39, -frac_line_37, frac_line_38, -frac_line_36};
fracture_6 = news; Plane Surface(fracture_6) = {frac_loop_6};
Physical Surface("FRACTURE_6") = {fracture_6};
Surface{fracture_6} In Volume{1};

Line{frac_line_62} In Surface{fracture_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_40, -frac_line_44, -frac_line_42, frac_line_43, -frac_line_41};
fracture_7 = news; Plane Surface(fracture_7) = {frac_loop_7};
Physical Surface("FRACTURE_7") = {fracture_7};
Surface{fracture_7} In Volume{1};

Line{frac_line_64} In Surface{fracture_7};

// Lines on in and outlet boundary
Line{in_line_low} In Surface{auxiliary_10};
Line{in_line_hi} In Surface{auxiliary_10};
Line{out_line_low} In Surface{auxiliary_11};
Line{out_line_hi} In Surface{auxiliary_11};


// End of fracture specification

// Start physical point specification
// End of physical point specification

