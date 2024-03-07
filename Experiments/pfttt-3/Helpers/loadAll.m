function allUsers = loadAll(basePath)
usersCount = length(dir(fullfile(basePath, 'user*')));
allUsers = cell(1, usersCount);
for i = 1:usersCount
    currentUser = load(getUserDir(basePath, i));
    allUsers{i} = currentUser;
end
end
