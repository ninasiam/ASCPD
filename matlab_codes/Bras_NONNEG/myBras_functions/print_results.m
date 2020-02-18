function [ ~ ] = print_results( in )
%     fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
%     semilogy([0:(size(MSE_Xiao1{i1},2)-1)],mean(MSE_Xiao1{i1},1),'-sb','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_Xiao2{i1},2)-1)],mean(MSE_Xiao2{i1},1),'-ob','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_Xiao3{i1},2)-1)],mean(MSE_Xiao3{i1},1),'->b','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_nina_accel{i1},2)-1)],mean(MSE_nina_accel{i1},1),'-xy','linewidth',1.5);hold on
% 
%     semilogy([0:(size(MSE_Xiao_adagrad{i1},2)-1)],mean(MSE_Xiao_adagrad{i1},1),'-dg','linewidth',1.5);hold on
%     legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','BrasCP accel','AdaCPD')
%     xlabel('no. of MTTKRP computed')
%     ylabel('MSE')
%     set(gca,'fontsize',14)
%     grid on
%     file_name = ['noise' '_' num2str(SNR(i1))];
%     saveas(fig1,[file_name '.pdf']);
%     
%     fig2 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
%     semilogy([0:(size(NRE_Xiao1{i1},2)-1)],mean(NRE_Xiao1{i1},1),'-sb','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_Xiao2{i1},2)-1)],mean(NRE_Xiao2{i1},1),'-ob','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_Xiao3{i1},2)-1)],mean(NRE_Xiao3{i1},1),'->b','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_nina_accel{i1},2)-1)],mean(NRE_nina_accel{i1},1),'-xy','linewidth',1.5);hold on
%     
%     semilogy([0:(size(NRE_Xiao_adagrad{i1},2)-1)],mean(NRE_Xiao_adagrad{i1},1),'-dg','linewidth',1.5);hold on
%     legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','BrasCP accel','AdaCPD')
%     xlabel('no. of MTTKRP computed')
%     ylabel('Relative Cost')
%     set(gca,'fontsize',14)
%     grid on
%     file_name = ['noise' '_' num2str(SNR(i1)) '_' 'cost'];
%     saveas(fig2,[file_name '.pdf']);


end

