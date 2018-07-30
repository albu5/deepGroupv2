annodir = 'my_anno';
annofmt = 'data_%2.2d.mat';

mkdir mask;
mask_fmt = 'mask/mask%2.2d.mat';

max_people = 10;

% format
% bbs(1:4), pose(5), action(6), group_members(7:26), pairwise_interaction(27:46), group_label, group_activity, scene_activity


for i = 1:33
    annostr = fullfile(annodir, sprintf(annofmt, i));
    anno = load(annostr);
    anno = anno.anno_data;
    
    n_people = numel(anno.people);
    n_frames = anno.nframe;
    
    mask_mat = ones(n_frames, n_people, n_people);
    for ped1 = 1:n_people     
       mask1 = (anno.people(ped1).action > 0);
       mask1 = mask1 & (anno.people(ped1).pose > 0);
       mask1 = mask1 & (sum(anno.people(ped1).bbs, 2)' > 0);
       mask1 = repmat(mask1', [1, n_people]);
       display(size(mask_mat(:, ped1, :)))
       display(size(mask1))
       mask_mat(:, ped1, :) = mask_mat(:, ped1, :) & ...
           reshape(mask1, [size(mask1, 1), 1, size(mask1, 2)]);
       mask_mat(:, :, ped1) = mask_mat(:, :, ped1) & mask1;
    end
    save(sprintf(mask_fmt, i), 'mask_mat');
    display(sprintf(mask_fmt, i))
end
