# Wang Bump with 11 Layers

algebraic3d
solid layer1 = orthobrick (-10, -10, 0; 10, 10, 0.0200) -bc=1;
solid layer2 = orthobrick (-10, -10, 0.0200; 10, 10, 0.0215) -bc=2;
solid layer3 = orthobrick (-10, -10, 0.0215; 10, 10, 0.0365) -bc=3;
solid layer4 = orthobrick (-10, -10, 0.0365; 10, 10, 0.0395) -bc=4;
solid layer5 = orthobrick (-10, -10, 0.0395; 10, 10, 0.0645) -bc=5;
solid layer6 = orthobrick (-10, -10, 0.0645; 10, 10, 0.0660) -bc=6;
solid layer7 = orthobrick (-10, -10, 0.0660; 10, 10, 0.0760) -bc=7;
solid layer8 = orthobrick (-10, -10, 0.0760; 10, 10, 0.0775) -bc=8;
solid layer9 = orthobrick (-10, -10, 0.0775; 10, 10, 0.0900) -bc=9;
solid layer10 = orthobrick (-10, -10, 0.0900; 10, 10, 0.0915) -bc=10;
solid layer11 = orthobrick (-10, -10, 0.0915; 10, 10, 0.1200) -bc=11;

tlo layer1;
tlo layer2;
tlo layer3;
tlo layer4;
tlo layer5;
tlo layer6;
tlo layer7;
tlo layer8;
tlo layer9;
tlo layer10;
tlo layer11;
