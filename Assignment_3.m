% MBF: Applied Quantitative Asset Management
% Spring 2025
%
% AQAM: Course Assignment
% 
%European Defense Sector Equity Fund
%-------------------------------------------------------------------------

clear; clc;
Assets = xlsread( 'C:\Users\emaho\Desktop\2 year lessons\AQAM\Assignment\assignment_prices.xlsx' );
Factors=csvread('C:\Users\emaho\Desktop\2 year lessons\AQAM\Assignment\Europe_3_FF_Factors.csv',1,1); 
Factors = log(1+Factors/100);
Factors = Factors(2:end,:);
Market=Factors( :, 1 );                         % Defines the market excess return
SMB=Factors( :, 2 );                            % Defines the Small-Minus-Big factor
HML=Factors( :, 3 );                            % Defines the High-Minus-Low factor
Rf = Factors(:,4);
No_Assets = size( Assets, 2 )-1;
Returns   = diff( log( Assets ) );
Returns(:, 12) = [];  % Remove 12th stock as it has too few observations


% Calculate Standard Statistics            %these 
Mean = mean(Returns,'omitnan');
Median = median(Returns);
Volatility = std(Returns);
Returns_for_corr = Returns;
Returns_for_corr(any(isnan(Returns_for_corr), 2), :) = [];
Corr = corr(Returns_for_corr);
Cov = cov(Returns_for_corr);

Skew = skewness(Returns);
Kurt = kurtosis(Returns);

Forecast_Variance = zeros(size(Returns, 2),1);
%Returns_Normalized = []

%Estimating Garch Model to use for VARCOV
Models_GARCH = cell(size(Returns, 2), 1);  % Pre-allocate cell array to store models

for i = 1:size(Returns,2)

    Returns_Help = Returns(:,i) - mean( Returns(:,i),'omitnan'); %here we just demean the returns, CER(?)
    %Returns_Normalized = [Returns_Normalized Returns_Help];

    mdl = garch(1,1);
    mdl = estimate(mdl, Returns_Help);
    %Fore = forecast(mdl,1,'Y0',Returns_Help)

    Models_GARCH{i} = mdl;                  % Store model in cell array



    %Forecast_Variance(i,1) = Fore

end;
% 
% Forecast_Volatility = sqrt(Forecast_Variance)
% 
% COV_GARCH = Forecast_Volatility * Forecast_Volatility' .*Correlation_Matrix_Returns

%Estimation of VARCOV in final period
for i = 1:size(Returns,2)
    Returns_Help = Returns(:,i) - mean( Returns(:,i),'omitnan');
    Returns_Help = Returns_Help(~isnan(Returns_Help));  % Remove NaN values
    Fore = forecast(Models_GARCH{i},1,'Y0',Returns_Help);
    Forecast_Variance(i,1) = Fore;
    
end
Forecast_Volatility = sqrt(Forecast_Variance);

VARCOV = Forecast_Volatility * Forecast_Volatility' .* Corr;

%Mean-Variance optimization
weights_MV = ones(No_Assets,1)/No_Assets;

% Objective function (Sharpe Ratio)
objective = @(weights_MV) -(Mean*weights_MV - Rf(end)) / sqrt(weights_MV' * VARCOV * weights_MV);  % Negative Sharpe Ratio to minimize

%Set constraints
A = []; b = [];  % No inequality constraints in this case\
Aeq = ones(1,No_Assets); beq = 1;  % weights sum to 1
lb = zeros(No_Assets,1); ub = 0.5 * ones(No_Assets,1);  % 0 ≤ weights ≤ 0.5

% Optimization
options = optimset('Display', 'off');
[w_MV, sr_MV] = fmincon(objective, weights_MV, A, b, Aeq, beq, lb, ub, [], options);

Er_MV = Mean * weights_MV;
Vol_MV = sqrt(weights_MV' * VARCOV * weights_MV);
SR_MV = -sr_MV * sqrt(12);   %annualize the SR