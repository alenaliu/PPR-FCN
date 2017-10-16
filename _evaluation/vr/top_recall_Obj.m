function  top_recall = top_recall_Obj(Nre, pred_conf_cell, pred_label_cell, pred_bb_cell, gt_label_cell, gt_bb_cell)
threshold = 0.5;

%load('gt.mat','gt_tuple_label','gt_obj_bboxes','gt_sub_bboxes');
num_imgs = length(pred_conf_cell);
for i=1:num_imgs
    [pred_conf_cell{i}, ind] = sort(pred_conf_cell{i},'descend'); %rank the object by conf score
    if length(ind) >= Nre
        pred_conf_cell{i} = pred_conf_cell{i}(1:Nre);
        pred_label_cell{i} = pred_label_cell{i}(ind(1:Nre));
        pred_bb_cell{i} = pred_bb_cell{i}(ind(1:Nre),:);
    else
        pred_label_cell{i} = pred_label_cell{i}(ind);
        pred_bb_cell{i} = pred_bb_cell{i}(ind,:);
    end
end

num_pos = 0;
for ii = 1 : num_imgs
    num_pos = num_pos + length(gt_label_cell{ii});
end
 

tp_cell = cell(1,num_imgs);
fp_cell = cell(1,num_imgs);

% iterate over images
for i=1:num_imgs 
 
    gt_Label = gt_label_cell{i};
    
    if ~isempty(gt_bb_cell{i})
        gt_box_entity = gt_bb_cell{i};
    else
        gt_box_entity = [];
    end
    
    num_gt = length(gt_Label);
    gt_detected = zeros(1,num_gt);
   
    labels = pred_label_cell{i};
    
    box = pred_bb_cell{i};
    if ~isempty(box)
        box_entity_our  = box;
    else
        box_entity_our  = [];
    end
    
    num_obj = length(labels);
    tp = zeros(1,num_obj);
    fp = zeros(1,num_obj);
    for j=1:num_obj

        bbO = box_entity_our(j,:); 
        ovmax = -inf;
        kmax = -1;
        
        for k=1:num_gt
            if labels(j) ~= gt_Label(k)
                continue;
            end
            if gt_detected(k) > 0
                continue;
            end
            
            bbgtO = double(gt_box_entity(k,:)); 
            
            biO=[max(bbO(1),bbgtO(1)) ; max(bbO(2),bbgtO(2)) ; min(bbO(3),bbgtO(3)) ; min(bbO(4),bbgtO(4))];
            iwO=biO(3)-biO(1)+1;
            ihO=biO(4)-biO(2)+1;
        
     
            
            if iwO>0 && ihO>0                
                % compute overlap as area of intersection / area of union
                uaO=(bbO(3)-bbO(1)+1)*(bbO(4)-bbO(2)+1)+...
                   (bbgtO(3)-bbgtO(1)+1)*(bbgtO(4)-bbgtO(2)+1)-...
                   iwO*ihO;
                ov =iwO*ihO/uaO;
                % makes sure that this object is detected according
                % to its individual threshold
                if ov >= threshold && ov > ovmax
                    ovmax=ov;% if the same detected bb overlaps with multiple gt, we only count the one with largest overlap.
                    kmax=k;
                end
            end
        end
        
        if kmax > 0
            tp(j) = 1;
            gt_detected(kmax) = 1;
        else
            fp(j) = 1;
        end
    end

    % put back into global vector
    tp_cell{i} = tp;
    fp_cell{i} = fp;

end

t = tic;
tp_all = [];
fp_all = [];
confs = [];
for ii = 1 : num_imgs
tp_all = [tp_all; tp_cell{ii}(:) ];
fp_all = [fp_all; fp_cell{ii}(:) ];
confs = [confs; pred_conf_cell{ii}(:)];
end

[confs, ind] = sort(confs,'descend');
tp_all = tp_all(ind);
fp_all = fp_all(ind); 

 
tp = cumsum(tp_all );
fp = cumsum(fp_all );
recall =(tp/num_pos);
top_recall = recall(end);

end