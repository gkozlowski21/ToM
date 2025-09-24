function [x] = betadist_plot(RP, betaA, betaB, model_acc, avg_action_prob)

    n = 6;
    % Generate x values for the beta distribution
    x = linspace(0, 1, 100);
    y = betapdf(x, real(betaA(n)), real(betaB(n)));

    % Check if the figure already exists, if not, create it
    figHandle = findobj('Type', 'Figure', 'Name', 'Beta Distribution');
    if isempty(figHandle)
        figure('Name', 'Beta Distribution'); % Create figure if it doesn't exist
    else
        figure(figHandle); % Use existing figure
    end
    
    % Clear current axes to update the plot
    cla; % Clear the current axes
    
    % Plot the beta distribution
    plot(x, y, 'b-', 'LineWidth', 2);
    hold on; % Keep the current plot
    
    % Add a vertical line at x = RP
    xline(RP(n), 'r--', 'LineWidth', 2);
    
    % Calculate y value at the intersection
    y_intersection = betapdf(RP(n), real(betaA(n)), real(betaB(n))); 
    
    % Add a label at the intersection
    text(RP(n), y_intersection, sprintf('y = %.2f', y_intersection), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
        'FontSize', 10, 'Color', 'red');
    
    % Format the plot
    grid on;
    xlabel('x');
    ylabel('Probability Density');
    title(['Beta Distribution (\alpha = ' num2str(real(betaA(n))) ', \beta = ' num2str(real(betaB(n))) ')']);
end


