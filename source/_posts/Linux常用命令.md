---
title: Linux常用命令
date: 2018-08-14 11:12:37
tags: [Linux]
categories: Linux
mathjax: true
---

### 1.常用指令 

> ls显示文件或目录 
>
> ​	-l列出文件详细信息(list) 
>
> ​	-a列出当前目录下所有文件及目录，包含隐藏的a(all) 
>
> mkdir创建目录 
>
> ​	-p创建目录，若无父目录，则创建p（parent） 
>
> cd切换目录 
>
> touch创建空文件 touch a.txt 
>
> echo创建带有内容的文件 echo ‘It is a good test’ >b.txt 
>
> cat查看文件内容 cat a.txt 
>
> cp拷贝文件 cp b.txt c.txt 
>
> mv移动或重命名 mv c.txt cc.txt 
>
> rm删除文件 rm cc.txt 
>
> ​	-r递归删除，可删除子目录及文件 
>
> ​	-f强制删除 
>
> find在文件系统中搜索某文件 find / -name 'hadoop-streaming' 
>
> wc统计文本中行数、字数、字符数 wc b.txt 
>
> grep在文本文件中查找某个字符串 grep ‘good’ b.txt 显示b.txt中包含good的行 
>
> rmdir删除空目录 
>
> tree树形结构显示目录 tree -a 
>
> pwd显示当前目录 pwd 
>
> In创建链接文件 
>
> more、less分页显示文本文件内容 more a.txt| less a.txt 
>
> head、tail显示文件头、尾内容 head a.txt| tail a.txt 

### 2.系统管理命令 

> stat显示指定文件的详细信息，比ls更详细 stat a.txt 
>
> who显示在线登陆用户 
>
> whoami显示当前操作用户 
>
> hostname显示主机名 
>
> uname显示系统信息 
>
> top动态显示耗费资源最多的进程信息 
>
> ps显示瞬间进程状态 
>
> du查看目录大小 du -h /home带有单位显示目录信息 
>
> df查看磁盘大小 df -h 带有单位显示磁盘信息 
>
> ifconfig查看网络情况 
>
> ping测试网络连通 
>
> netstat显示网络状态信息 
>
> man查看命令 man ping查看ping信息 
>
> clear清屏 
>
> kill杀死进程 

### 3.打包压缩相关命令 

> tar:打包压缩 
>
> ​    -c归档文件 
>
> ​    -x压缩文件 
>
> ​    -z gzip压缩文件 
>
> ​    -j bzip2压缩文件 
>
> ​    -v显示压缩或解压缩过程v(view) 
>
> ​    -f使用档名 
>
> tar -cvf Test.tar Test只打包，不压缩 
>
> tar -zcvf Test.tar.gz Test打包并使用gzip压缩 
>
> tar -jcvf Test.tar.bz2 Test打包并使用bzip2压缩 
>
> 解压缩只需要将上述c替换成x便可以 

### 4.关机/重启机器 

> shutdown 
>
> ​    -r关机重启 
>
> ​    -h关机不重启 
>
> ​    Now立刻关机 
>
> halt关机 
>
> reboot重启 

### 5.Vim简单实用

> vim使用 
>
> vim三种模式：命令模式、插入模式、编辑模式。使用ESC或i来切换模式。 
>
> 命令模式如下 
>
> :q退出 
>
> :q！强制退出 
>
> :wq保存并退出 
>
> :set number显示行号 
>
> :set nonnumber隐藏行号 

### 6.推广

更多内容请关注公众号**谓之小一**，若有疑问可在公众号后台提问，随时回答，欢迎关注，内容转载请注明出处。

![推广](Linux常用命令/推广.png)