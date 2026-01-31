## 一、git指令等

- #### （一）使用信息记录

  **git --version**
  <u>git version 2.45.1.windows.1</u>
  git config --**global user.name**
  git config --**global user.email**

- #### （二）路径

  cd C:\Users\你的用户名\项目文件夹
  cd /c/Users/你的用户名/项目文件夹
- #### （三）首次上传本地项目文件夹到github对应的仓库的流程

  **1.**  在网页**<u>新建空仓库</u>**（不要勾选add readme和其他添加默认初始文件的选项）
  **2.**  右键电脑<u>**本地项目文件夹**</u>，选中 "**<u>git bash here</u>**"，会弹出git bash窗口
  **3.**  进入git bash，输入指令序列
  `$ git init  #初始化
  $ git remote add origin 你的github空仓库网址
  $ git add .
  $ git commit -m "提交说明"
  $ git checkout -b main
  $ git push -u origin main`
  注意**仓库中必须已有文件**，`git add .`才能上传新的文件，**`git push -u origin main`** **才能正常运行**

- #### （四）更新说明

  `git add .
  git commit -m "更新说明"
  git push`

- （五）从github仓库上抓取到本地ok
  k