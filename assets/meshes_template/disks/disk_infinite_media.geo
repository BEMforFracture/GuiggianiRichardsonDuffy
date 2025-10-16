// Gmsh project created on Wed Aug 20 17:19:14 2025
SetFactory("OpenCASCADE");
//+
Disk(1) = {0, -0, -0, 1, 1};
//+
Curve Loop(2) = {1};
//+
Plane Surface(2) = {2};
//+
Physical Curve("F", 3) = {1};
//+
Physical Surface("C", 4) = {1};