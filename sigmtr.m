function [p_tr]=sigmtr(p,lb,ub,tau)
p_tr=lb+(ub-lb)./(1+exp(-p./tau));
end