function [ print ] = print_results( SNR, in, bottlenecks )
    
    if(strcmp(in.constraint,'unconstraint'))
        error_rbs = in.cpdrbs;
    else
        error_full = in.full;
    end 
    error_bras_opt = in.bras_opt;
    error_bras_accel = in.bras_accel;
    error_adagrad = in.ada;
    error_1 = in.bras_1;
    
    fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
    if(strcmp(in.constraint,'unconstraint'))
        semilogy([0:(size(error_rbs,2)-1)],mean(error_rbs,1),'->b','linewidth',1.5);hold on
        name_l = 'RBS'; 
    else
        semilogy([0:(size(error_full,2)-1)],mean(error_full,1),'->b','linewidth',1.5);hold on
        name_l = 'AO NTF';
    end
    
    semilogy([0:(size(error_1,2)-1)],mean(error_1,1),'-sb','linewidth',1.5);hold on
    semilogy([0:(size(error_bras_opt,2)-1)],mean(error_bras_opt,1),'-ob','linewidth',1.5);hold on
    semilogy([0:(size(error_bras_accel,2)-1)],mean(error_bras_accel,1),'->g','linewidth',1.5);hold on
    semilogy([0:(size(error_adagrad,2)-1)],mean(error_adagrad,1),'-xy','linewidth',1.5);hold on
    
    legend( name_l, 'Bras 0.1','BrasCPD optimal step','BrasCP accel','AdaCPD')
    xlabel('no. of n MTTKRP computed')
    ylabel('Relative error')
    set(gca,'fontsize',14)
    grid on

    if(strcmp(bottlenecks,'on'))
        if(strcmp(in.constraint,'unconstraint'))
            file_name = ['./figures_24_2/uncon/bottlecks/' in.path 'bottlecks' '_' num2str(1)];
            saveas(fig1,[file_name '.pdf']);
        else
            file_name = ['./figures_24_2/nonneg/bottlecks/' in.path 'bottlecks' '_' num2str(1)];
            saveas(fig1,[file_name '.pdf']);
        end
    else
        if(strcmp(in.constraint,'unconstraint'))
            file_name = ['./figures_24_2/uncon/noise/14_3/' in.path 'noise' '_' num2str(SNR)];
            saveas(fig1,[file_name '.pdf']);
        else
            file_name = ['./figures_24_2/nonneg/noise/' in.path 'noise' '_' num2str(SNR)];
            saveas(fig1,[file_name '.pdf']);
        end
    end
    print = 'printing';
    disp('Figures are saved at . directory');
end

