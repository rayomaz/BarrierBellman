% Plot barrier function

clc; clear; close all;

plot_barrier_2D()
find_minimum()

print('barrier.pdf', '-dpdf', '-r300');

function plot_barrier_2D()

    x1 = linspace(-1, 1, 100);

    % Bounds on polynomial kernel
    f = 1.0000007651953218*x1.^2 - 1.7197334168473105e-11*x1 - ...
        9.088427152842553e-10;

    f2 = 0.2200384646464848*ones(1, length(x1));

    eta = 0.04;

    plot(x1, f, 'LineWidth', 3)
    hold on
    plot(x1, f2, 'LineWidth', 3)

    xline(-0.2,'k--','LineWidth',2);
    xline(+0.2,'k--','LineWidth',2);
    xline(-1.0,'r--','LineWidth',2);
    xline(+1.0,'r--','LineWidth',2);
    yline(eta,'m--','LineWidth',2)
    text(-.05,0.5,'$X_0$','Interpreter','latex','FontSize',16)
    text(+1.05,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-1.20,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-0.90,0.1,['$\eta = $', num2str(eta)],'Interpreter','latex',...
        'Color', 'magenta', 'FontSize',16)
    xlabel('x');
    ylabel('B(x)');
    grid on
    ylim([-.1, 2.5])
    xlim([-1.25, 1.25])
    labels = {'\color{blue} SOS B(x)'};
    legend(labels, 'Location', 'NorthWest', 'FontSize', 8, ...
    'TextColor', 'black');

end

function find_minimum()

    % Barrier function
    f = @(x1) 1.0000007651953218*x1.^2 - 1.7197334168473105e-11*x1 - ...
        9.088427152842553e-10;

    % Find the minimum of the function in the interval [-10,10]
    [x_min, f_min] = fminbnd(f,-10,40);
    
    % Display the result
    fprintf(['The minimum value of the function is f(x)=%f,' ...
        ' which occurs at x=%f.\n'],f_min,x_min);

end


