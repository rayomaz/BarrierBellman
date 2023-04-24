% Plotting ibp bounds on erf functions

clc; clear; close;

load("linearsystem_5.mat")
% add PiecewiseBarrier/tests/partitions/test/ folder to path 

% Probability distribution on hyperspace
figure
hold on
grid on

% Notice: this example is for transition from Xj to Xi
% where i = j = 1

for jj = 1:length(upper_partition)
    
    colors = {'b', 'k', 'm', 'r', 'c'};
    for ii = 1:1%length(upper_partition)
    
        x_space = linspace(lower_partition(jj), upper_partition(jj), 1000);
        
        % Bounds
        A_low = lower_probability_bounds_A(ii, jj, 1, :);
        b_low = lower_probability_bounds_b(ii, jj, :);

        A_up = upper_probability_bounds_A(ii, jj, 1, :);
        b_up = upper_probability_bounds_b(ii, jj, :);

        prob_bound_lower = A_low * x_space + b_low;
        prob_bound_upper = A_up * x_space + b_up;

        % True erf
        prob_true = zeros(1, length(x_space));
        sigma = 0.1;
        epsilon = 1e-6;
        m = 1;      % sys dim
        const = 1/(2^m);
        vl = lower_partition(ii);
        vu = upper_partition(ii);
        for pp = 1:length(x_space)
            y = 0.95*x_space(pp);
            erf_low = (y - vl)/(sigma*sqrt(2));
            erf_up = (y - vu)/(sigma*sqrt(2));
            prob_true(pp) = const*(erf(erf_low) - erf(erf_up));
        end

        plot(x_space, prob_bound_lower, "LineWidth", 3, 'color', "k")
        plot(x_space, prob_bound_upper, "LineWidth", 3, 'Color', "b")

        plot(x_space, prob_true, "LineWidth", 3, 'Color', "r")
    
    end
end

labels = {'\color{black} lower bound', '\color{blue} upper bound', ...
    '\color{red} true'};
legend(labels, 'Location', 'NorthEast', 'FontSize', 8, ...
'TextColor', 'black');





