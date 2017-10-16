for i=0:69
    
    
    recall50P = top_recall_Relationship_all(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours,i);
    fprintf('%d %.2f\n',i, recall50P);
    %disp(strcat(int2str(i),'  df'));
end