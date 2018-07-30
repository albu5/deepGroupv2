annodir = 'my_anno';
annofmt = 'data_%2.2d.mat';

intdir = 'pairwise_int';
intfmt = 'int%2.2d.mat';

grpdir = 'group_int';
grpfmt = 'int%2.2d.mat';

mkdir csv_anno;
csvfmt = 'csv_anno/anno%2.2d.mat';

max_people = 10;

% format
% bbs(1:4), pose(5), action(6), group_members(7:26), pairwise_interaction(27:46), group_label, group_activity, scene_activity

attrs_tensor = zeros(n_people, n_people, pair_feat_len, n_frames);

for i = 1:33
    annostr = fullfile(annodir, sprintf(annofmt, i));
    anno = load(annostr);
    anno = anno.anno_data;
 
    n_people = numel(anno.people);
    n_frames = anno.nframe;
    
    pair_feat_len = 2 * (4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1);
    % bbs:4 | action | pose | in_grp | pair_interaction | grp_label |
    % grp_activity| scene_activity | mask
    % only in_grp and pair_interaction are pairwise quantities
    
    for t = 1:anno.nframe
        for ped1 = 1:n_people
           for ped2 = 1:n_people
               bbs1 = anno.people(ped1).bbs(t, :);
               action1 = anno.people(ped1).action(t);
               pose1 = anno.people(ped1).pose(t);
               
               in_grp = anno.groups.grp_label(t, ped1) == anno.groups.grp_label(t, ped1);
               in_grp = in_grp & (anno.groups.grp_label(t, ped1) > 0);
               
               pair_int = pint(ped1, ped2, t);
               grp_label1 = anno.groups.grp_label(t, ped1);
               grp_act11 = anno.groups.grp_act(t, ped1);
               scene_act = anno.Collective(t);
               mask1 = (action1 > 0) && (pose1 > 0);
               
               bbs2 = anno.people(ped2).bbs(t, :);
               action2 = anno.people(ped2).action(t);
               pose2 = anno.people(ped2).pose(t);
               grp_label2 = anno.groups.grp_label(t, ped2);
               grp_act12 = anno.groups.grp_act(t, ped2);
               mask2 = (action2 > 0) && (pose2 > 0);
               feat_vec = [bbs1, action1, pose1,...
                           in_grp, pair_int,...
                           grp_label1, grp_act1,...
                           scene_act1, mask1,...
                           bbs2, action2, pose2,...
                           in_grp, pair_int,...
                           grp_label2, grp_act2,...
                           scene_act2, mask2];
                attrs_tensor(ped1, ped2, :, t) = feat_vec;
           end
        end
    end
    
    for ped = 1:n_people
        ped_attr = anno.people(ped).bbs;
        ped_attr = horzcat(ped_attr, anno.people(ped).pose');
        ped_attr = horzcat(ped_attr, anno.people(ped).action');
        
        pad_mat = zeros(size(group_int, 3), max_people-n_people);
        ped_attr = horzcat(ped_attr, squeeze(group_int(ped, :, :))', pad_mat);
        
        pad_mat = zeros(size(pint, 3), max_people-n_people);
        ped_attr = horzcat(ped_attr, squeeze(pint(ped, :, :))', pad_mat);
        
        grp_label_ind = anno.groups.grp_label(:, ped);
        grp_act = zeros(size(grp_label_ind));
        for gi = 1:numel(grp_act)
            if grp_label_ind(gi) > 0
                grp_act(gi) = anno.groups.grp_act(gi, grp_label_ind(gi));
            end
        end
        
        ped_attr = horzcat(ped_attr, anno.groups.grp_label(:, ped));
        ped_attr = horzcat(ped_attr, grp_act);
        ped_attr = horzcat(ped_attr, anno.Collective');
                
        if numel(peds_attrs) == 0
            peds_attrs = ped_attr;
        else
            peds_attrs = cat(3, peds_attrs, ped_attr);
        end
    end
    save(sprintf(csvfmt, i), 'peds_attrs');
    display(sprintf(csvfmt, i))
end
