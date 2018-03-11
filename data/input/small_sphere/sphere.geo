algebraic3d

solid cube = plane (-0.3, -0.3, 0; 0, 0, -1)
         and plane (-0.3, -0.3, 0; 0, -1, 0)
         and plane (-0.3, -0.3, 0; -1, 0, 0)
         and plane (0.3, 0.3, 0.1; 0, 0, 1)
         and plane (0.3, 0.3, 0.1; 0, 1, 0)
         and plane (0.3, 0.3, 0.1; 1, 0, 0);
solid sph = sphere (0.0, 0.0, 0.02; 0.01);

solid rest = cube and not sph;

tlo rest -transparent -col=[0,0,1];
tlo sph -col=[1,0,0];

