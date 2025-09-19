clc
close all
clear
%%
U_grid = 10:10:40;
ER1_grid4 = 0.8:0.04:1;
ER1_grid3 = 0.74:0.04:1;
ER1_grid2 = 0.68:0.04:1;
ER1_grid1 = 0.6:0.04:1;

% d=100, h=260
ER1 =[0.57   0.62    0.66    0.72     0.78    0.83];
U1 = [9.442  15.659  24.707  35.195  45.553   54.054];
ER1O =[0.66      0.66    0.73     0.8    0.84  0.89];% Oscillation
U1O = [8.27     14.77   22.44     32   42.35  50.41];
% d=100, h=90
ER2 =[0.56  0.63  0.68  0.74  0.8  0.85];
U2 = [7.5  15.5  24.3  34.5  44.5  52.7];

% d=88, h=260
ER3 =[0.54   0.61   0.73    0.78   0.82    ];
U3 = [8.76  15.99   40.94   51.18  59.68   ];
ER3O =[0.6   0.65    0.75      0.8      0.84];% Oscillation
U3O = [8      15     40        49.74   58.47];
% d=88, h=90
ER4 =[0.82];
U4 = [46.57];

% d=120, h=90
ER6 =[0.79];
U6 = [42.62];

figure
scatter(ER1, U1, 100, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w',"LineWidth",1.5);hold on
scatter(ER1O, U1O, 100, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w',"LineWidth",1.5);hold on
scatter(ER2, U2, 100, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k',"LineWidth",1.5);hold on
scatter(ER3, U3, 100, 's', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w',"LineWidth",1.5);hold on
scatter(ER3O, U3O, 100, 'p', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w',"LineWidth",1.5);hold on
scatter(ER4, U4, 100, 's', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b',"LineWidth",1.5);hold on
scatter(ER6, U6, 100, 'd', 'MarkerEdgeColor', 'm', 'MarkerFaceColor', 'm',"LineWidth",1.5);hold on
plot(ER1_grid1, U_grid(1)*ones(size(ER1_grid1)), 'ro', 'DisplayName','U=10');hold on
plot(ER1_grid2, U_grid(2)*ones(size(ER1_grid2)), 'ro', 'DisplayName','U=20');hold on
plot(ER1_grid3, U_grid(3)*ones(size(ER1_grid3)), 'ro', 'DisplayName','U=30');hold on
plot(ER1_grid4, U_grid(4)*ones(size(ER1_grid4)), 'ro', 'DisplayName','U=40');hold on

legend('D=100 mm, H=260 mm', 'Oscillation','D=100 mm, H=90 mm','D=88 mm, H=260 mm','Oscillation','D=88 mm, H=90 mm','D=120 mm, H=90 mm','Location','best')
xlabel('$\it\Phi$',Interpreter='latex');xlim([0.2 1])
ylabel('$U$ [m/s]',Interpreter='latex')
set(gca,"LineWidth",1,'FontName','Times new roman','fontsize',16)