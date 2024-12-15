# 1. 配置环境
- 安装webotsR2023b版本
- 打开webots，在webots_world文件夹下选择对应的世界文件
- 在webots_controller文件夹下选择对应的leader控制器
- 设置follower控制器为extern
- 
# 2. 配置测试脚本

- 测试之前需要修改配置文件：env_config_x.yaml，主要修改测试world名字为相应的world名字，
- 在env_config_x配置相应参数，如use_random_mask等。
# 启动脚本
- 在终端输入命令：
```
python ppo_test.py
```
