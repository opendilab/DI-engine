Git 使用
~~~~~~~~~~~~~~~~

如果对 git 几乎不了解，那么推荐看一下\ `廖雪峰 git 教程 <https://www.liaoxuefeng.com/wiki/896043488029600>`__\ ，会对相关概念和命令有一个正确且充分的认识。下面罗列一下实际开发中常用的知识。


1. 基础概念和命令
^^^^^^^^^^^^^^^^^^

原始代码、工作区、暂存区、本地仓库、远程仓库的\ **概念**\ ，和它们之间\ **互相转换的命令**\ ：

.. image:: ./images/git_command1.png
    :scale: 33%
    :align: center

下面介绍一些没有在上图展示，但也很常用的命令：

**git stash命令**\ ：如果正在开发分支 A，此时突然来了个工作需要切换到分支 B，而 A 又还没有到可以提交一个 commit 的程度，就可以使用\ ``git stash``\ 先将对分支 A 的修改保存起来（如果希望像commit一样留下一些信息，可以使用\ ``git stash save "STASH-MESSAGE"``\ ）。
然后通过 ``git checkout xxx`` 命令切换到分支 B（如果没有执行\ ``git stash``\ 或\ ``git commit``\ 则会报错）。在分支 B 施工完成后，可以切换回分支 A，然后利用\ ``git stash pop``\ 将暂存的内容恢复。

stash 是一个栈式结构，如果需要 pop 某个非栈顶元素，可以先使用\ ``git stash list``\ 查看所有的 stash 记录，然后使用命令\ ``git stash pop stash@{0}``\ ，其中 0 可以替换为任何存在的 stash 记录编号。

.. image:: ./images/git_command2_stash.png
    :scale: 33%
    :align: center

**git log命令**\ ：可以显示提交 commit 的信息，可参考\ `教程 <https://www.yiibai.com/git/git_log.html>`__

**git cherry-pick命令**\ ：可以将某个 commit 应用到其它的分支上，其和 git merge 的区别是，git merge 会将整个分支合并进其它分支，
而 git cherry-pick 只会将某个 commit 应用在其它分支，可参考\ `教程 <https://ruanyifeng.com/blog/2020/04/git-cherry-pick.html>`__


2. Git Commit 规范
^^^^^^^^^^^^^^^^^^^^^^^

我们对 git commit 进行了一些规定：

1. 尽量每一个独立的功能对应一个 commit

2. | 模板：feature/fix/polish/test/style(commiter_name or project_name):
     commit message
   | 举例：fix(zlx): add tb logger in naive buffer


3. 以一次实际开发为例
^^^^^^^^^^^^^^^^^^^^^^^^^^^

下面以一次实际开发过程为例，讲解需要用到的 git 命令。

1. 克隆仓库：\ ``git clone REPO-URL``\ ；

2. 切换到自己的开发分支：\ ``git checkout -b YOUR-BRANCH-NAME``

3. 进行一些修改，每完成一个功能后，就将希望提交的内容\ ``git add``\ 进来（这里可以先利用\ ``git status``\ 查看所有的改动，如果想添加所有修改的文件，可以使用\ ``git add -u``\ 命令）；然后提交一个commit：\ ``git commit -m COMMIT-MESSAGE``\ 。

4. 将本地仓库推送到远程仓库：\ ``git push origin YOUR-BRANCH-NAME``\ 。如果是首次推送，会提示远程仓库没有与之关联的分支，按照提示修改命令即可，一般为\ ``git push --set-upstream origin YOUR-BRANCH-NAME``\ 。如果在你提交本次修改前，别人也修改了代码并推送到了远程仓库，会提示存在冲突，需要利用\ ``git pull origin YOUR-BRANCH-NAME --rebase``\ 命令将最新的代码拉取下来，并解决最新代码与自己的代码之间的冲突（若有），然后才可push。

5. 分支合并命令：\ ``git rebase BRANCH-NAME``\ 。在我们的开发中，如果单独切出分支并提了 pull request，则必须保证该分支可以无冲突地合并进 main。故 merge 命令常常使用于以下场景：A 同学与 B 同学分别从 main 分支切出 C 分支和 D 分支进行开发，A 同学完成了 C 分支并将其 merge 进了 main 分支，B 同学在开发的最后，需要\ ``git rebase main``\ 并解决全部冲突，才算是这个分支开发结束。


4. .gitignore 文件
^^^^^^^^^^^^^^^^^^^^^

我们本地的开发路径下，有很多不想提交到远程仓库的文件，比如项目的本地配置信息、pycache、log文件、checkpoint等等。这时，使用 .gitignore 文件可以通过字符匹配的方式忽略掉这些文件，就可以更加愉快地使用\ ``git add .``\ 或\ ``git add -u``\ 命令了（当然，此时还是需要先\ ``git status``\ 查看一下都增加/删除/修改了哪些文件）。

.gitignore 文件中常见的写法如下：

.. code:: 

   # 忽略指定文件
   HelloWrold.class
   # 忽略指定文件夹
   pkg/
   __pycache__/
   # *是通配符，可以匹配任何字符串
   # 忽略所有 .jpg 文件
   *.jpg
   # 忽略名称中末尾为ignore的文件夹
   *ignore/
   # 忽略名称中间包含ignore的文件夹
   *ignore*/

如果确实想要上传某些被 .gitignore 规则拦截的文件（比如图片）可以使用\ ``git push -f PATH``\ 来上传。但请务必保证这个文件是必要的。

