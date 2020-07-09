
function plot_boundary_2(A, y, x, method, fig)
    
    label_1 = find(y == 1);
    label_0 = find(y == 0);

    fig = plot(A(label_1,1), A(label_1,2), 'k+', 'LineWidth', 2, 'Markersize', 7);
    hold on;
    plot(A(label_0,1), A(label_0,2), 'ko', 'MarkerFaceColor', 'y', 'Markersize', 7);
    hold on;
    
    plot_x = [min(A(:,1))-5,  max(A(:,1))+5];
    plot_y = (-1./x(2)).*(x(1).*plot_x + 0);
    plot(plot_x, plot_y);
    hold off;
    
    grid on;
    xlabel('a_1')
    ylabel('a_2')
    title("Data")
    legend('Class 1','Class 0',method);
    axis([-10, 10, -10, 10])
  
end