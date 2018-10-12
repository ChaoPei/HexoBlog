---
title: Git常用命令总结
date: 2018-09-13 16:07:11
tags: [Git]
categories: Git
mathjax: false
---

很早之前就有使用Git来维护项目，但都是基于个人开发，所以开发维护起来也就很方便，使用的Git命令也很少。但工作之中，需要考虑各个成员之间的合作，所以就需要增加Git技能。刚好经过一段时间的学习，在这儿总结一些相关命令。

首先我们需要了解

\### 1.创建版本库 

mkdir test //创建新文件夹 

cd test 

git init //变成git可管理的仓库 

**创建一个新文件** 

vim readme.txt 

\> test 

git add readme.txt //添加到仓库 

git commit -m "add readme.txt" //备注添加readme.txt信息 

\### 版本回退 

git log// 查看历史提交信息 

git reflog //记录历史每次命令 

git log --pretty=oneline //只显示重要历史信息 

git reset --hard 8e19abd40a //回到历史某个版本 

git reset既可以回退版本，也可以回退掉暂存区的内容。 

\### 管理修改 

git diff HEAD -- readme.txt //查看工作区和版本去的区别 

git checkout -- readme.txt //撤销修改 

场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令git checkout -- file。此时还没有提交 

场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令git reset HEAD <file>，就回到了场景1，第二步按场景1操作。 

场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，进行版本回退。 

git rm readme.txt //删除文件 

\### 远程仓库 

github创建仓库test 

git remote add origin <https://github.com/weizhixiaoyi/weizhixiaoyi.github.io.git> //将本地关联到远端 

git push -u origin master //提交到远端，并将本地和远端关联起来。以后提交可以省略-u。 

git push origin master //提交到远程master分支 

git push origin zhenhai //提交到远程zhenhai分支 

git push -f origin master //强制性将数据传送到远程 

git clone <https://github.com/weizhixiaoyi/test.git//>从远程库克隆到本地 

git clone -b zhenhai <https://github.com/weizhixiaoyi/test.git> //从远程仓库克隆某个分支到本地 

\### 分支管理 

**增加删除** 

git branch //查看分支 

git checkout -b dev //创建并切换分支 

git checkout -b zhenhai origin/online-master//创建dev分支，并依赖online-master分支进行构建 

git push origin zhenhai:zhenhai //将本地分支更新到远程，这样远程也有dev分支 

git branch --set-upstream-to=origin/zhenhai //将本地分支关联到远程 

git branch -d dev //删除本地分支 

git branch -D dev //强项删除本地分支 

b 

git push origin --delete dev //删除远程名为dev的分支 

git checkout master //切换回master分支 

git merge dev //切换到master分支后，合并dev分支 

git branch -a //查看远程所有分支 

git log --graph //看分支状态图 

git log --graph --pretty=oneline --abbrev-commit //查看分支提交详细信息 

**更新合并分支** 

git push origin zhenhai //将当前分支提交到zhenhai分支 

git pull origin zhenhai:zhenhai //从远程仓库zhenhai分支更新到本地zhenhai分支。 

git merge dev //合并dev-1.1分支。首先回到dev分支，然后git merge dev-1.1 

git merge --no-ff -m "merge with no-ff" dev //保存分支历史 

当我们想要看到分支情况是，使用git merge --no--ff。 

![a15da4e5b4a8487f1efa835b68be1fb5.png](<evernotecid://28F5AE6A-EC38-4FEB-8611-36B8FB4E1295/appyinxiangcom/17484315/ENNote/p358?hash=a15da4e5b4a8487f1efa835b68be1fb5>) 

**bug分支** 

当出现bug时候，我们需要紧急去修复，但当前分支并没有完成，还不能进行提交。可以将当前分支信息进行暂存起来，然后基于此分支构建bug分支，当bug修复成功之后，进行合并到该分支之上。然后再恢复以前缓存的信息。 

git stash //暂存当前工作去，放心的创建bug分支去修复bug。 

git stash list //查看已暂存的stash缓存。 

git stash apply //恢复储存的stash。但恢复之后，stash并没有删除，需要利用git stash drop进行删除。 

git stash apply stash@{0} // 指定恢复特定的缓存。 

git stash pop //恢复的同时把stash也进行删除。 

**feature分支** 

每新增加一个功能，最后新创建一个分支，新功能在分支上测试完成之后，再合并到主分支上。 

当正在开发的功能被抛弃时，可以使用 

git branch -d feature-dev来进行删除分支。 

删除出现问题的话，可以使用git branch -D feature-dev删除分支。 

**多人开发合作** 

master上有dev分支，你正在基于dev分支做一些开发。 

然后团队另外一个小伙伴clone了master分支，并创建了dev 分支进行开发，开发完成之后并进行了提交。 

当你在dev分支上进行开发，开发完成之后进行提交，却发现提交失败。提示说明远方dev分支比本地新，所以我们需要将远程dev分支拉取过来(git pull)，然后重新提交。当然建议使用git pull —rebase，可以少一次提交，使得版本控制更加清晰。 

当再次进行提交时，发现可能会有冲突，可以参考上面提到的冲突解决的方法，冲突解决之后便可提交到远程库。 



**git rebase** 

git pull --rebase //合并分支，隐式合并分支，不产生多余commit。 

**git merge和git rebase的区别** 

（1）使用git merge合并分支，解决完冲突，执行add和commit操作，此时会产生一个额外的commit。如下图： 

![4058fb8b60273da160a4261114d119e4.jpeg](<evernotecid://28F5AE6A-EC38-4FEB-8611-36B8FB4E1295/appyinxiangcom/17484315/ENNote/p358?hash=4058fb8b60273da160a4261114d119e4>) 

（2）使用git rebase合并分支，解决完冲突，执行add和git rebase --continue，不会产生额外的commit。这样master分支上不会有无意义的commit。如下图： 

所以可以这么说：merge是显性合并，rebase是隐性合并。 

同理，当你执行git pull时，是同时执行了git fetch 和 git merge两个操作。如果你不想进行merge操作，即不想留下合并的记录，可以使用 git pull --rebase操作。 

![1f1e1c9c1e457bb29387167be5599624.jpeg](<evernotecid://B62BFF83-0265-4C5E-8C49-235081AF7ED3/appyinxiangcom/17484315/ENResource/p2740>) 



**标签管理** 

开发过程中，每次完成一个新版本功能，我们可以给此版本打上标签。将来在任何时候，取某个标签版本，就是把标签时刻的历史版本拿过来。所以，标签也是版本的一个快照。 

Git之中打标签也相当方便，我们只需要切换到相应分支，然后利用git tag version便可。例如git tag v1.0。也可以对历史的某个版本进行打上标签，例如git tag v1.1 f52c633。可以通过git tag -a v1.0 -m “tag new version” 1094ab来对标签进行增加描述。 

使用git tag可以查看所有的标签。使用git show version可以查看标签信息，例如git show v1.0。 



某些情况下可能会打错标签，可以使用git tag -d v1.0来进行删除标签。如果标签已经推送到远程，可能就比较麻烦，首先我们需要在本地进行删除标签，git tag -d v1.0，然后从远程也进行删除，git push origin :refs/tags/v1.0。 



如果已经打好标签，想要推送到远程，可以使用git push origin v1.0。或者一次性推送所有以前未推送的标签，git push origin —tags。
 



**自定义git** 

比如可以让git显示相关颜色，git config - -global color.ui true。 

忽略特殊文件：我们可以创建.git里面创建一个.gitignore的文件，然后把需要忽略提交的文件名称放在里面就行了。 

1. 忽略操作系统自动生成的文件，比如缩略图等； 
2. 忽略编译生成的中间文件、可执行文件等，也就是如果一个文件是通过另一个文件自动生成的，那自动生成的文件就没必要放进版本库，比如Java编译产生的.class文件； 
3. 忽略你自己的带有敏感信息的配置文件，比如存放口令的配置文件。 















在实际开发中，我们应该按照几个基本原则进行分支管理： 

首先，master分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活； 

那在哪干活呢？干活都在dev分支上，也就是说，dev分支是不稳定的，到某个时候，比如1.0版本发布时，再把dev分支合并到master上，在master分支发布1.0版本； 

你和你的小伙伴们每个人都在dev分支上干活，每个人都有自己的分支，时不时地往dev分支上合并就可以了。 

所以，团队合作的分支看起来就像这样![3fde99592e66bb0030802f7f77bbd57e.png](<evernotecid://28F5AE6A-EC38-4FEB-8611-36B8FB4E1295/appyinxiangcom/17484315/ENResource/p2668>) 

通常，合并分支时，如果可能，Git会用Fast forward模式，但这种模式下，删除分支后，会丢掉分支信息。 

如果要强制禁用Fast forward模式，Git就会在merge时生成一个新的commit，这样，从分支历史上就可以看出分支信息。 

准备合并dev分支，请注意--no-ff参数，表示禁用Fast forward： 

$ git merge --no-ff -m "merge with no-ff" dev 

Merge made by the 'recursive' strategy. 

readme.txt | 1 + 

1 file changed, 1 insertion(+) 

**创建独立分支** 

git checkout --orphan orphan-dev //创建单独分支 


































