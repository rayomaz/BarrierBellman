% Plot barrier function

clc; clear; close all;


% Probability distribution on hyperspace
figure
hold on
grid on

partitions = [10, 40, 100, 500, 1000];

x = linspace(-1.0, 1.0, 1000);
barrier_sos = 1.0000009348300702*x.^2 - 2.9799150000775737e-12*x + ...
              4.114065673172966e-7;

plot(x, barrier_sos, "LineWidth", 4, "Color", 'r');

text_to_print = append("β (SOS) = ", string(0.01));
text(-0.2, 1.15, text_to_print, 'Color', 'r');

colors = ['b', 'g', 'c', 'm', 'k'];

for kk = 1:length(partitions)

    % Load relevant data
    file_data    = append("linearsystem_", string(partitions(kk)));
    file_barrier = append("barrier_", string(partitions(kk)));
    load(file_data)
    load(file_barrier)

    hypercubes = length(upper_partition);

    for jj = 1:hypercubes

        x_space = linspace(lower_partition(jj), upper_partition(jj), 1000);
    
        constant_barrier = b(jj)*ones(1,length(x_space));
        
        plot(x_space, constant_barrier, "LineWidth", 4, "Color", colors(kk));
    end
    
    beta_round = round(beta(end) * 1000) / 1000;
    text_to_print = append("β (K = ", string(partitions(kk)), " ) = ",...
                           string(beta_round));
    text(-0.2, kk/length(partitions), text_to_print,'Color', colors(kk));

    clear -partitions -colors

end

% 10: Solution: [η = 1.0e-6, β = 0.16274889004599408]
% 20: Solution: [η = 1.0e-6, β = 0.06965642799317558]
% 30: Solution: [η = 1.0e-6, β = 0.03486097848488312]
% 40: Solution: [η = 1.0e-6, β = 0.020079872497465283]
% 50: Solution: [η = 1.0e-6, β = 0.012970892306800595]
% 60: Solution: [η = 1.0e-6, β = 0.009014996844327229]
% 70: Solution: [η = 1.0e-6, β = 0.0066506061530683174]
% 80: Solution: [η = 1.0e-6, β = 0.005133020345203534]
% 90: Solution: [η = 1.0e-6, β = 0.004113689316897768]
% 100: Solution: [η = 1.0e-6, β = 0.0033962054913270565]
% 200: Solution: [η = 1.0e-6, β = 0.0012843395584788775]
% 500: Solution: [η = 1.0e-6, β = 0.0007305252511220767]

% SOS
% B(x) = 1.0000009348300702*x^2 - 2.9799150000775737e-12*x + 4.114065673172966e-7
% Solution: [η = 9.957064809533035e-7, β = 0.010000010857703032]





