初始化要求
Python版本3.12


拉取代码
git checkout main
git pull

开发时要创建分支（不准直接上传main）
# 创建并切换到新分支，名字规范 feature/姓名-功能
git checkout -b feature/marui-data-import
其实分支随便命名

在分支上提交代码
git add .
git commit -m "feat: 完成数据导入基础逻辑"


在分支上同步main的代码（就是在你前一个pull之前可能其他人又有提交了）
# 把main最新代码同步到你的分支
git pull --rebase origin main

推送自己的分支到 GitHub
git push origin feature/marui-data-import

经过审核后会合并到main

最后可以删除分支
# 切回主分支同步最新代码
git checkout main
git pull
# 删除本地废弃功能分支
git branch -d feature/marui-data-import