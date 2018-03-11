algebraic3d

solid cube = plane (-0.3, -0.3, 0; 0, 0, -1)
         and plane (-0.3, -0.3, 0; 0, -1, 0)
         and plane (-0.3, -0.3, 0; -1, 0, 0)
         and plane (0.3, 0.3, 0.1; 0, 0, 1)
         and plane (0.3, 0.3, 0.1; 0, 1, 0)
         and plane (0.3, 0.3, 0.1; 1, 0, 0);
solid sph = sphere (0.008, 0.0, 0.01; 0.005);
solid sph2 = sphere (-0.008, 0.0, 0.03; 0.005);

solid test = ellipsoid (0.0,0.0,0.02; 0.005196152422707,0,-0.003000000000000; 0,0.01,0; 0.0085,0,0.014722431864335);

solid rest = cube and not test and not sph and not sph2;

tlo rest -transparent -col=[0,0,1];

tlo sph -col=[1,0,0];

tlo sph2 -col=[1,0,0];

tlo test -col=[1,0,0];
