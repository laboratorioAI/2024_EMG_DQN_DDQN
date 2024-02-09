function filePath = getUserDir(baseDir, idx)
    folders = dir(fullfile(baseDir, 'user*'));
    if idx < 1 || idx > numel(folders)
        error('Index out of range.');
    end
    dirName = folders(idx).name;
    filePath = fullfile(baseDir, dirName, 'userData.mat');
end
