close all;
X = 100;
gamma = 0.99;
A = 2;
p = 0.5;
q_low = 0.51;
q_high = 0.6;
q = [q_low, q_high];
P=getuncontrolled(X,A, p, q_high,q_low);
p_lazy=lazy_policy(X,A);
p_aggr=aggr_policy(X,A);
r=get_reward(X,A);
F=pl_feature(X);
maxiter=100;
maxiter_lstd=1e+5;
eta=0.1;
M=100;
eta_array=logspace(-2,2,M);
rewards=zeros(M,1);
for i=1:M
    rewards(i)=soft_policy_iter(F,X,A,r,p,q,P, maxiter,maxiter_lstd,gamma, eta_array(i));
    fprintf("eta round: %d", i)
end
[argvalue, argmax]=max(rewards);
opt_eta = eta_array(argmax)
semilogx(eta_array, rewards)
xline(opt_eta,color="r")
ylabel("total rewards")
xlabel("eta values")
title("Rewards from soft policy iteration")









function [P] = getuncontrolled(X,A, p, q_high,q_low)
    P = zeros(X,X,A);
    for i=1:X
        P(i, i, 1) = (1 - p) * (1 - q_low) + p * q_low;
        P(i, i, 2) = (1 - p) * (1 - q_high) + p * q_high;
        if i + 1 < X+1
            P(i, i + 1, 1) = p * (1 - q_low) ; % 0 corresponds to low action
            P(i, i + 1, 2) = p * (1 - q_high);  % 1 corresponds to high action
        end
        if i - 1 > 0
            P(i, i - 1, 1) = q_low * (1 - p);  % 0 corresponds to low action
            P(i, i - 1, 2) = q_high * (1 - p);  % 1 corresponds to high action
        end
        if i == 1
            P(i, i, 1)=P(i, i, 1)+q_low * (1 - p) ; % 0 corresponds to low action
            P(i, i, 2) = P(i, i, 2) + q_high * (1 - p) ; % 0 corresponds to low action
        end
        if i == X
            P(i, i, 1) = P(i, i, 1) + p * (1 - q_low) ; % 0 corresponds to low action
            P(i, i, 2) = P(i, i, 2) + p * (1 - q_high) ; 
        end
    end
end

function [p]=lazy_policy(X,A)
    p = zeros(X,A);
    for i=1:X
        p(i,1)=1;
    end
end

function [p]=aggr_policy(X,A)
    p = zeros(X,A);
    for i=1:X
        if i<=50
            p(i,1)=1;
        else
            p(i,2)=1;
        end
    
    end
end
function [p]=uniform_policy(X,A)
    p = zeros(X,A);
    for i=1:X
        for j=1:A
            p(i,A)=0.5;
        end
    
    end
end

function [r] = get_reward(X,A)
    r=zeros(X,A);
    for i=1:X
        for j=1:A
            c=cost(j);
            r(i,j)= -(i/X)^2 - c;
        end
    end
end

function [c] = cost(a)
    c=0;
    if a==2
        c=0.05;
    end
end

function [F] =coarse_feature(X)
    F=zeros(X/5,X);
    for k=1:X/5
        i = k;
        j = 5 * (i - 1)+1;
        while j <=( 5 * i )
            if j>=1 &&  j<= X
                F(k, j) = 1;
            end
            j=j+ 1;
        end
    end
end

function [F] = pl_feature(X)
    F=zeros(2*X/5,X);
    F_coarse = coarse_feature(X);
    F(1:X/5,:) = F_coarse;
    for k=1:X/5
        i = k;
        j = 5 * (i - 1)+1;
        while j <= 5 * i
            if j>=1 && j <= X
                F((X/5)+k, j) = (j-1 - 5 * (i - 1)) / 5;
            end
            j = j + 1;
        end
    end
end
function [p] = trans(pol,P,X)
    p=zeros(X,X);
    for i=1:X
        for j=1:X
            P_new=[P(i,j,1),P(i,j,2)];
            p(i,j)=dot(P_new,pol(i,:));
        end
    end
end

function [V,s,total_reward]=lstd(policy,P, F,s0, maxiter,X,A,gamma)
    sigma = 1e-5;
    D = size(F,1);
    A_mat = eye(D)+sigma;
    B_mat = zeros(D,1);
    reward=get_reward(X,A);
    total_reward=0;
    theta=zeros(X,1);
    p_pi=trans(policy,P,X);
    s=s0;
    states=1:X;
    for i=1:maxiter
        [argvalue, argmax] = max(policy(s,:));
        r = reward(s,argmax);
        next_s= randsample( states, 1, true, p_pi(s,:) );
        v = F(:,s)- gamma*F(:,next_s);
        A_mat = A_mat +F(:,s)* v.';
        B_mat = B_mat + r * F(:, s);
        s=next_s;
        theta=inv(A_mat)*B_mat;
        total_reward=total_reward+r;

    end
    V= (theta.'*F).';

end


function [total_reward]=soft_policy_iter(F,X,A,r,p,q,P, maxiter,maxiter_lstd,gamma, eta)
    Q_est=zeros(X,A);
    policy=uniform_policy(X,A);
    new_policy=zeros(X,A);
    total_reward=0;
    s0=100;
    for i=1:maxiter
        [V,s0,r_lstd]=lstd(policy,P, F,s0, maxiter_lstd,X,A,gamma);
        total_reward=total_reward+r_lstd;
        for x=1:X
            for a=1:A
                if x==1
                    l=x;
                    u=x+1;
                elseif x==X
                    l=x-1;
                    u=x;
                else
                    l=x-1;
                    u=x+1;
                end
               
                Q_est(x, a) = r(x, a) + gamma * (1 - p) * (q(a) * V(l) + (1 - q(a)) * V(x))+ gamma * p * (q(a) * V(x) + (1 - q(a)) * V(u));
               

            end
            [argvalue, argmax]=max(Q_est(x,:));
            Q_est(x, :) = Q_est(x, :) - argvalue;
            for a=1:A
                new_policy(x, a) = policy(x, a) * exp(eta * Q_est(x, a));
            end
            new_policy(x, :) = new_policy(x, :) / sum(new_policy(x, :));
        end
        policy=new_policy;
    fprintf("soft policy iteration: %d\n", i)
    end
end
