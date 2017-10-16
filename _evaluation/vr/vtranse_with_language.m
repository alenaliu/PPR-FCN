%   This file is for Predicting <subject, predicate, object> phrase and relationship

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors",
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

%% data loading
addpath('evaluation');
load('data/objectListN.mat'); 
% given a object category index and ouput the name of it.

load('data/obj2vec.mat'); 
% word-to-vector embeding based on https://github.com/danielfrg/word2vec
% input a word and ouput a vector.

load('data/UnionCNNfea.mat'); 
% the CNN score on union of the boundingboxes of the two participating objects in that relationship. 
% we provide our scores (VGG based) here, but you can re-train a new model.

load('data/objectDetRCNN.mat');
% object detection results. The scores are mapped into [0,1]. 
% we provide detected object (RCCN with VGG) here, but you can use a better model (e.g. ResNet).
% three items: 
% detection_labels{k}: object category index in k^{th} testing image.
% detection_bboxes{k}: detected object bounding boxes in k^{th} testing image. 
% detection_confs{k}: confident score vector in k^{th} testing image. 

load('data/Wb.mat');
% W and b in Eq. (2) in [1]

testNum = 1000;
fprintf('#######  Relationship computing Begins  ####### \n');
for i = 1 : length(rlp_labels_ours)
    
    if mod(i, 100) == 0
        fprintf([num2str(i), 'th image is tested! \n']);
    end
    for ii=1:size(rlp_labels_ours{i})
        %disp(ii);
        %disp(rlp_labels_ours{i}(ii,1));
        sub_vec = obj2vec(objectListN{rlp_labels_ours{i}(ii,1)});
        obj_vec = obj2vec(objectListN{rlp_labels_ours{i}(ii,3)});
        vec_org  = [sub_vec,obj_vec,1];
        languageModual =  [W,B]*vec_org';
        r_vec = relation_vectors{i}(ii,:)+5.5;
        %rlpScore = (1./(1+exp(-languageModual'))).*r_vec;
        %rlpScore = tanh(languageModual').*r_vec;
        rlpScore = languageModual'.*r_vec;
        [m_score, m_preidcate]  = max(rlpScore); 
        %rlp_labels(ii,2)=m_preidcate;
        rlp_labels_ours{i}(ii,2)=m_preidcate-1;
        
        %rlp_confs_ours{i}(ii)=rlp_confs_ours{i}(ii)*m_score;
        %sub_bboxes_ours{i}(ii,1)=0;
    end

end

%% sort by confident score
for ii = 1 : length(rlp_confs_ours)
    [Confs, ind] = sort(rlp_confs_ours{ii}, 'descend');
    rlp_confs_ours{ii} = Confs;
    rlp_labels_ours{ii} = rlp_labels_ours{ii}(ind,:);
    sub_bboxes_ours{ii} = sub_bboxes_ours{ii}(ind,:);
    obj_bboxes_ours{ii} = obj_bboxes_ours{ii}(ind,:);
end

recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours); 
fprintf('%0.2f \n', 100*recall100P);
fprintf('%0.2f \n', 100*recall50P);

recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('%0.2f \n', 100*recall100R);
fprintf('%0.2f \n', 100*recall50R);



fprintf('\n');
fprintf('\n');
zeroShot100P = zeroShot_top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50P = zeroShot_top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('%0.2f \n', 100*zeroShot100P);
fprintf('%0.2f \n', 100*zeroShot50P);

zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('%0.2f \n', 100*zeroShot100R);
fprintf('%0.2f \n', 100*zeroShot50R);