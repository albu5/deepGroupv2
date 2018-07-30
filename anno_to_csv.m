annodir = '/data/my_anno';
annofmt = 'data_%2.2d.mat';

intdir = 'pairwise_int';
intfmt = 'int%2.2d.mat';

mkdir csv_anno;
csvfmt = 'csv_anno/anno%2.2d.mat';

max_people = 20;

% format
% bbs(1:4), pose(5), action(6), group_members(7:26), pairwise_interaction(27:46), group_label, group_activity, scene_activity

for i = 1:33
    annostr = fullfile(annodir, sprintf(annofmt, i));
    anno = load(annostr);
    anno = anno.anno_data;
    
    intstr = fullfile(intdir, sprintf(intfmt, i));
    pint = load(intstr);
    pint = pint.interaction;
    
    n_people = numel(anno.people);
    group_int = zeros([n_people, n_people, anno.nframe]);
    
    peds_attrs = [];
    
    for t = 1:anno.nframe
        % form group belonging matrix
        group_label = anno.groups.grp_label(t,:);
        group_label = repmat(group_label, [n_people, 1]);
        group_int(:,:,t) = group_label == group_label';
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
