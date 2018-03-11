algebraic3d

solid slab = orthobrick (-0.3, -0.3, 0; 0.3, 0.3, 0.04) -bc=1;
solid ellip = ellipsoid (0.0, 0.0, 0.02; 0.02, 0,0.0; 0, 0.01, 0; 0.0,0,0.01);
solid rest = slab and not ellip;

tlo rest -transparent -col=[0,0,1];
tlo ellip -col=[1,0,0];