function run_fastspm_batch(input_file, output_dir, spm_dir, recipe_file)
% Run the packaged FastSPM Unified Segmentation recipe on one T1 image.
%
% The input must be an uncompressed NIfTI file. The driver writes only the
% modulated, normalized GM map (mwc1*) to output_dir; SPM's other temporary
% tissue files remain in the temporary working directory created by the shell
% wrapper and are removed after a successful run.

if nargin < 3 || isempty(spm_dir)
    spm_dir = getenv('SPM12_DIR');
end
if nargin < 4 || isempty(recipe_file)
    recipe_file = fullfile(fileparts(mfilename('fullpath')), 'fastspm_v1.m');
end
if isempty(spm_dir)
    error('SPM12_DIR is not set');
end
addpath(spm_dir);
spm('defaults', 'fmri');
spm_jobman('initcfg');

if ~exist(input_file, 'file')
    error('Input image does not exist: %s', input_file);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if ~exist(recipe_file, 'file')
    error('FastSPM recipe does not exist: %s', recipe_file);
end
run(recipe_file);
matlabbatch{1}.spm.spatial.preproc.channel.vols = {input_file};
spm_jobman('run', matlabbatch);

[input_dir, input_name, input_ext] = fileparts(input_file);
expected = fullfile(input_dir, ['mwc1' input_name input_ext]);
if ~exist(expected, 'file')
    candidates = dir(fullfile(input_dir, 'mwc1*.nii'));
    if isempty(candidates)
        error('SPM completed without producing an mwc1 map for %s', input_file);
    end
    expected = fullfile(input_dir, candidates(1).name);
end

output_file = fullfile(output_dir, [input_name '_mwc1.nii']);
if exist(output_file, 'file')
    delete(output_file);
end
if ~copyfile(expected, output_file)
    error('Could not copy FastSPM output to %s', output_file);
end
fprintf('FastSPM output: %s\n', output_file);
end
