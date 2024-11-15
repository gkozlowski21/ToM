function [p]=invsigmtr(p_tr,lb,ub,tau)
p=-log((ub-p_tr)./(p_tr-lb));
p=p*tau;
end