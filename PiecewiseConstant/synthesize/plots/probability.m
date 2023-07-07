% Plotting ibp-interval bounds on erf functions

clc; clear; close;

% Probability distribution on hyperspace
partitions = load('../models/linear/state_partitions.txt');
data_hyper = load('../models/linear/probability_data_5_sigma_0.01.mat');
prob_lower = data_hyper.matrix_prob_lower;
prob_upper = data_hyper.matrix_prob_upper;

figure
hold on
grid on

% Note: this example is for transition from Xj to Xi

for jj = 1:length(partitions)
    
    colors = {'b', 'k', 'm', 'r', 'c'};
    for ii = 3%1:1%length(partitions)
    
        x_space = linspace(min(partitions(jj, :)), max(partitions(jj, :)), 1000);
        
        % True erf
        prob_true = zeros(1, length(x_space));
        sigma = 0.01;
        epsilon = 1e-6;
        m = 1;      % sys dim
        const = 1/(2^m);

        vl = min(partitions(ii, :));
        vu = max(partitions(ii, :));

        % Prod of erfs
        for pp = 1:length(x_space)
            y = 0.95*x_space(pp);
            erf_low = (y - vl)/(sigma*sqrt(2));
            erf_up = (y - vu)/(sigma*sqrt(2));
            prob_true(pp) = const*(erf(erf_low) - erf(erf_up));
            % Notice this computes Pu --> Pu = 1 - Ps
        end

        min_prob_exact = prob_lower(jj, ii);
        max_prob_exact = prob_upper(jj, ii);

        plot(x_space, prob_true, "LineWidth", 3, 'Color', "r")

        plot(x_space, min_prob_exact*ones(1, length(x_space)), "LineWidth", 3, 'Color', "b")
        plot(x_space, max_prob_exact*ones(1, length(x_space)), "LineWidth", 3, 'Color', "k")
 
    end
end

labels = {'\color{red} true', '\color{blue} lower bound', ...
          '\color{black} upper bound'};
legend(labels, 'Location', 'NorthEast', 'FontSize', 8, ...
'TextColor', 'black'); 
xlabel("$x$", 'Interpreter','latex', "FontSize", 25)
ylabel('$P$', 'Interpreter','latex', "FontSize", 25)
title("Probability distribution")
set(gcf,'color','w');
set(gca,'FontSize',14)



