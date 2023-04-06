% Plot barrier function

clc; clear; close all;

plot_barrier_2D()
find_minimum()

% Save plots
% print('barrier.pdf', '-dpdf', '-r300');
print('barrier_2.png', '-dpng', '-r300');


function plot_barrier_2D()

    x1 = linspace(-5, -3, 100);

    % Bounds on polynomial kernel
    f = 1.000000311720917*x1.^2  + 8.000002181494006*x1 +...
        16.00000373885447;

    eta = 0.04;
        
    plot(x1, f, 'LineWidth', 3)
    hold on
    xline(-4.2,'k--','LineWidth',2);
    xline(-3.8,'k--','LineWidth',2);
    xline(-5.0,'r--','LineWidth',2);
    xline(-3.0,'r--','LineWidth',2);
    yline(eta,'m--','LineWidth',2)
    text(-4.05,0.5,'$X_0$','Interpreter','latex','FontSize',16)
    text(-2.95,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-5.25,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-4.90,0.1,['$\eta = $', num2str(eta)],'Interpreter','latex',...
        'Color', 'magenta', 'FontSize',16)
    xlabel('x');
    ylabel('B(x)');
    grid on
    xlim([-5.5, -2.5])
    ylim([-.1, 1.5])
    labels = {'\color{blue} SOS B(x)'};
    legend(labels, 'Location', 'NorthWest', 'FontSize', 8, ...
    'TextColor', 'black');

end

function find_minimum()

    % Barrier function
    f = @(x1) 1.000000311720917*x1.^2  + 8.000002181494006*x1 +...
        16.00000373885447;

    % Find the minimum of the function in the interval [-10,10]
    [x_min, f_min] = fminbnd(f,-10,40);
    
    % Display the result
    fprintf(['The minimum value of the function is f(x)=%f,' ...
        ' which occurs at x=%f.\n'],f_min,x_min);

end


