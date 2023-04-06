% Plot barrier function

clc; clear; close all;

plot_barrier_2D()
find_minimum()

% Save plots
% print('barrier.pdf', '-dpdf', '-r300');
print('barrier_3.png', '-dpng', '-r300');


function plot_barrier_2D()

    x1 = linspace(13, 15, 100);

    % Bounds on polynomial kernel
    f = 0.0001544781659944353*x1.^2 - 0.00425860951633615*x1 + ...
        1.029455051630625;
        
    plot(x1, f, 'LineWidth', 3)
    hold on
    xline(13.8,'k--','LineWidth',2);
    xline(14.2,'k--','LineWidth',2);
    xline(13.0,'r--','LineWidth',2);
    xline(15.0,'r--','LineWidth',2);

    text(13.95,1.0002,'$X_0$','Interpreter','latex','FontSize',16)
    text(12.75,1.0002,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(15.05,1.0002,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    xlabel('x');
    ylabel('B(x)');
    grid on
    xlim([12.5, 15.5])
    labels = {'\color{blue} SOS B(x)'};
    legend(labels, 'Location', 'NorthWest', 'FontSize', 8, ...
    'TextColor', 'black');

end

function find_minimum()

    % Barrier function
    f = @(x1) 0.0001544781659944353*x1.^2 - 0.00425860951633615*x1 + ...
        1.029455051630625;

    % Find the minimum of the function in the interval [-10,10]
    [x_min, f_min] = fminbnd(f,-10,40);
    
    % Display the result
    fprintf(['The minimum value of the function is f(x)=%f,' ...
        ' which occurs at x=%f.\n'],f_min,x_min);

end


