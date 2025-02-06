# GitHub GPU 训练项目

这是一个使用GitHub Actions进行GPU模型训练的示例项目。该项目展示了如何利用GitHub提供的免费GPU资源进行深度学习模型的训练。

## 项目结构

```
github-gpu-training/
├── .github/
│   └── workflows/
│       └── train.yml    # GitHub Actions 工作流配置
├── src/
│   └── train.py        # 训练脚本
├── tests/              # 测试目录
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

## 功能特点

- 使用GitHub Actions进行自动化训练
- 利用免费的GPU资源
- 自动保存训练模型
- 支持模型训练结果的下载

## 使用方法

1. Fork 这个仓库到你的GitHub账号
2. 启用GitHub Actions（在仓库的Settings -> Actions中）
3. 推送代码到main分支或创建pull request会自动触发训练
4. 也可以手动在Actions标签页中触发工作流

## 训练结果

训练完成后，模型文件会被自动保存并上传为构建产物。你可以在GitHub Actions的运行记录中下载训练好的模型。

## 注意事项

- GitHub Actions的免费额度对公共仓库是无限的
- 每个作业最长运行时间为6小时
- 建议在代码中加入检查点保存功能，以防训练中断

## 许可证

MIT 