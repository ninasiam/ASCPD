function fig = plot_data(A, y)

    label_1 = find(y == 1);
    label_0 = find(y == 0);

    fig = plot(A(label_1,1), A(label_1,2), 'k+', 'LineWidth', 2, 'Markersize', 7);
    hold on;
    plot(A(label_0,1), A(label_0,2), 'ko', 'MarkerFaceColor', 'y', 'Markersize', 7);
    grid on;
    xlabel('a_1')
    ylabel('a_2')
    title("Data")
    axis([-10, 10, -10, 10])
    hold off;
end

