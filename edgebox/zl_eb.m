function bbs = zl_eb(in_path,alpha,beta,minscore,maxboxes)
    global model
    global opts
    opts.alpha = alpha;
    opts.beta = beta;
    opts.minScore = minscore;
    opts.maxBoxes = maxboxes;
    I = imread(in_path);
    bbs=edgeBoxes(I,model,opts);

    bbs = [bbs(:,1) bbs(:,2) bbs(:,1)+bbs(:,3) bbs(:,2)+bbs(:,4) bbs(:,5)];
end
