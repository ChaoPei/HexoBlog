---
title: Mac+Hexo+GitHub博客搭建教程
date: 2018-03-16 11:41:35
tags: [Mac,Hexo,GitHub,博客]
categories: 教程
toc: true
comments: true
---

### 1.为什么写博客

以前利用Jekyll+Github搭建博客，但每次博客搭建完成后都没有继续坚持写博文，直到最近找实习才认识到技术博客的重要性。以前学习的很多知识点都已经忘记啦，所以下定决心这次认真总结以前学习的知识点，认真写点技术文章。

### 2.Mac+Hexo+GitHub博客

现在博客主流的就是Jekyll和Hexo两种格式，选择Jekyll还是Hexo就根据个人喜好啦，但个人更推荐使用Hexo，选择Hexo的主要原因。

+ Jekyll没有本地服务器，无法实现本地文章预览，需要上传到WEB容器中才能预览功能，而Hexo可以通过简单的命令实现本地预览功能，并直接发布到WEB容器中实现同步。
+ Jekyll主题和Hexo主题对比而言，Hexo主题更加简洁美观(个人审美原因)。

选择GitHub的原因不用多说，程序员的乐园，更是支持pages功能，虽然很多其他社区也支持，比如GitLab、coding、码云等，但GitHub更加活跃，自己的项目就是放在上面，所以更加方便。但GitHub有最大一点不好之处便是*百度爬虫无法爬去博客内容*，自己也找了好久解决方法，比如利用coding托管(免费版绑定域名有广告)、CDN加速(对于流量太小的网站没什么用)，所以暂时没什么太好的解决方法。

### 3.博客本地环境搭建

#### 3.1安装Node.js和Git

Mac上安装可以选择图形化方式和终端安装，此处直接使用客户端方式安装。Node.js官网下载文件，根据提示安装即可，安装成功后在目录*/usr/local/bin*目录下。测试Node.js和npm，出现下述信息则安装成功。

```
node -v
v8.10.0
```

```
npm -v
5.6.0
```

Git官网下载相应文件根据提示直接进行安装，检查git是否安装成功，直接查看git版本即可。

> Git --version 
>
> git version 2.15.0

#### 3.2安装Hexo

Node.js和Git都安装成功后开始安装Hexo。安装时注意权限问题，加上sudo，其中-g表示全局安装。

```mac
sudo npm install -g hexo
```

#### 3.3博客初始化

创建存储博客的文件，比如命名为myblog，然后进入到myblog之中。

```
cd myblog
```

执行下述命令初始化本地博客，下载一些列文件。

```
hexo init
```

执行下述命令安装npm。

```
sudo npm install
```

执行下述命令生成本地html文件并开启服务器，然后通过http://localhost:4000查看本地博客。

```
hexo g
hexo s
```

![图片3.3](Mac+Hexo+GitHub博客搭建教程/图片3.3.png)

### 4.本地博客关联GitHub

#### 4.1本地博客代码上传GitHub

注册并登陆GitHub账号后，新建仓库，名称必须为`user.github.io`，如`weizhixiaoyi.github.io`。

![图片01](Mac+Hexo+GitHub博客搭建教程/图片4.1.png)

终端cd到myblog文件夹下，打开_config.yml文件。或者用其他文本编辑器打开可以，推荐sublime。

```Vim
vim _config.yml
```

打开后至文档最后部分，将deploy配置如下。

```Python
deploy:
  type: git
  repository: https://github.com/weizhixiaoyi/weizhixiaoyi.github.io.git
  branch: master
```

其中将repository中`weizhixiaoyi`改为自己的用户名，注意type、repository、branch后均有空格。通过如下命令在myblog下生成静态文件并上传到服务器。

```
hexo g
hexo d
```

若执行`hexo g`出错则执行`npm install hexo --save`，若执行`hexo d`出错则执行`npm install hexo-deployer-git --save `。错误修正后再次执行`hexo g`和`hexo d`。

若未关联GitHub，执行`hexo d`时会提示输入GitHub账号用户名和密码，即:

```
username for 'https://github.com':
password for 'https://github.com':
```

`hexo d`执行成功后便可通过https://weizhixiaoyi.github.io访问博客，看到的内容和http://localhost:4000相同。

#### 4.2添加ssh keys到GitHub

添加ssh key后不需要每次更新博客再输入用户名和密码。首先检查本地是否包含ssh keys。如果存在则直接将ssh key添加到GitHub之中，否则进入新生成ssh key。

执行下述命令生成新的ssh key，将`your_email@example.com`改成自己以注册的GitHub邮箱地址。默认会在`~/.ssh/id_rsa.pub`中生成`id_rsa`和`id_rsa.pub`文件。

```
ssh-keygen -t rsa -C "your_email@exampl"		
```

Mac下利用`open ~/.ssh  `打开文件夹，打开id_rsa.pub文件，里面的信息即为ssh key，将此信息复制到GitHub的Add ssh key`路径GitHub->Setting->SSH keys->add SSH key`界面即可。Title里填写任意标题，将复制的内容粘贴到key中，点击Add key完成添加。

此时本地博客内容便已关联到GitHub之中，本地博客改变之后，通过`hexo g`和`hexo d`便可更新到GitHub之中，通过https://weizhixiaoyi.github.io访问便可看到更新内容。

### 5.更换Hexo主题

可以选择Hexo主题官网页面搜索喜欢的theme，这里我选择hexo-theme-next当作自己主题，hex-theme-next主题是GitHub中hexo主题star最高的项目，非常推荐使用。

终端cd到myblog目录下执行如下所示命令。

```
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

将blog目录下_config.yml里的theme的名称`landscape`更改为`next`。

执行如下命令（每次部署文章的步骤）

```
hexo g  //生成缓存和静态文件
hexo d  //重新部署到服务器
```

当本地博客部署到服务器后，网页端无变化时可以采用下述命令。

```
hexo clean  //清楚缓存文件(db.json)和已生成的静态文件(public)
```

### 6.配置Hexo-theme-next主题

Hexo-theme-next主题便为精于心、简于形，简介的界面下能够呈现丰富的内容，访问[next官网](http://theme-next.iissnan.com/)查看配置内容。配置文件主要修改next中_config.yml文件，next有三种主题选择，分别为Muse、Mist、Pisces三种，个人选择的是Pisces主题。主题增加标签、分类、归档、喜欢（书籍和电影信息流）、文章阅读统计、访问人数统计、评论等功能，博客界面如下所示。

![图片6.1](Mac+Hexo+GitHub博客搭建教程/图片6.1.png)

![图片6.1](Mac+Hexo+GitHub博客搭建教程/图片6.2.png)

![图片6.1](Mac+Hexo+GitHub博客搭建教程/图片6.3.png)

#### 6.1增加标签、分类、归档页

首先将next/config.yml文件中将`menu`中`tags` ` catagories` `archive`前面的`#`。例如增加标签页，通过`hexo new page 'tags'`增加新界面，在myblog/sources中发现多了tags文件夹，修改index.md中内容，将type更改为`tags`。利用`hexo g`和`hexo d`将界面重新上传到服务器便可看到新增加的标签页，分类和归档页同理。

#### 6.2增加‘喜欢’界面

‘喜欢’界面用于展现自己看过的书籍和电影。

安装